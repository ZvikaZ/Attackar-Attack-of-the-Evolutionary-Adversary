import torch.nn.functional as F
from operator import itemgetter
import numpy as np
import random
import torch

from utils import normalize, plt_torch_image
from l2_utils import meta_pseudo_gaussian_pert

device = 'cuda' if torch.cuda.is_available() else 'cpu'

elitism_enabled = False  # TODO


def elitism(cur_pop):
    elite = min(cur_pop, key=itemgetter(1))[0]
    return elite


class EvoAttack():
    def __init__(self, dataset, model, x, y, n_gen=500, pop_size=40, eps=0.3, tournament=35, defense=False,
                 norm='linf', count=None):
        self.dataset = dataset
        self.model = model
        self.x = x
        self.y = y
        self.p_init = 0.1
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.n_tournament = tournament
        self.eps = eps
        self.best_x_hat = None
        self.queries = 0
        self.defense = defense
        self.min_ball = torch.tile(torch.maximum(self.x - eps, torch.tensor(0)), (1, 1))
        self.max_ball = torch.tile(torch.minimum(self.x + eps, torch.tensor(1)), (1, 1))
        self.norm = norm
        self.count = count

    def generate(self):
        gen = 0
        cur_pop = self.pop_init()
        while not self.termination_condition(cur_pop, gen):
            self.fitness(cur_pop)
            new_pop = []

            if elitism_enabled:
                elite = elitism(cur_pop)
                new_pop.append([elite, np.inf])

            if self.norm == 'linf':
                for i in range((self.pop_size - 1) // 3):
                    parent1 = self.selection(cur_pop)
                    parent2 = self.selection(cur_pop)
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                    mut_offspring1 = self.mutation((offspring1, np.inf))
                    mut_offspring2 = self.mutation((offspring2, np.inf))
                    offspring1 = self.project(offspring1)
                    new_pop.append([offspring1, np.inf])
                    new_pop.append([mut_offspring1, np.inf])
                    new_pop.append([mut_offspring2, np.inf])
            elif self.norm == 'l2':
                # for l2 we do only mutation, without crossover
                for i in range(self.pop_size - int(elitism_enabled)):  # we might have already added 1 elitist
                    ind, _ = self.selection(cur_pop)
                    mut_ind = self.mutation((ind, np.inf))
                    plt_torch_image(ind, self.x, f'Evo {self.count}: Gen {gen}, ind {i}')
                    new_pop.append([mut_ind, np.inf])
            else:
                raise ValueError(f'Unrecognized norm: {self.norm}')

            # print(f'Elitist: {min(cur_pop, key=itemgetter(1))[1]:.5f}')

            cur_pop = new_pop
            gen += 1

        return {'x_hat': self.best_x_hat, 'queries': self.queries, 'gen': gen + 1}

    def crossover(self, parent1, parent2):
        parent1 = parent1[0].flatten()
        parent2 = parent2[0].flatten()
        i = np.random.randint(0, len(parent1))
        j = np.random.randint(i, len(parent1))
        offspring1 = torch.cat([parent1[:i], parent2[i:j], parent1[j:]], dim=0)
        offspring2 = torch.cat([parent2[:i], parent1[i:j], parent2[j:]], dim=0)
        offspring1 = offspring1.reshape(self.x.shape)
        offspring2 = offspring2.reshape(self.x.shape)
        offspring1 = self.project(offspring1)
        offspring2 = self.project(offspring2)
        return offspring1, offspring2

    def selection(self, cur_pop):
        selection = [random.choice(cur_pop) for i in range(self.n_tournament)]
        best = min(selection, key=itemgetter(1))
        return best

    def mutation(self, x_hat):
        if self.norm == 'l2':
            return self.l2_mutation(x_hat)
        elif self.norm == 'linf':
            return self.linf_mutation(x_hat)
        else:
            raise ValueError(f'Unrecognized norm: {self.norm}')

    def l2_mutation(self, x_hat):
        min_val, max_val = 0, 1
        p = self.p_selection(self.p_init, self.queries, self.n_gen * self.pop_size)
        c = x_hat[0].shape[1]
        h = x_hat[0].shape[2]
        w = x_hat[0].shape[3]
        n_features = c * h * w

        x_curr = x_hat[0]
        delta_curr = self.x - x_curr

        s = max(int(round(np.sqrt(p * n_features / c))), 3)

        if s % 2 == 0:
            s += 1

        s2 = s + 0

        ### window_1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        new_deltas_mask = torch.zeros(x_curr.shape).to(device)
        new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

        ### window_2
        center_h_2 = np.random.randint(0, h - s2)
        center_w_2 = np.random.randint(0, w - s2)
        new_deltas_mask_2 = torch.zeros(x_curr.shape).to(device)
        new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0
        ## commented because it's not used:
        # norms_window_2 = torch.sqrt(
        #     torch.sum(delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] ** 2, axis=(-2, -1),
        #            keepdims=True))

        ### compute total norm available
        curr_norms_window = torch.sqrt(
            torch.sum(((self.x - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
        curr_norms_image = torch.sqrt(torch.sum((self.x - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
        mask_2 = torch.maximum(new_deltas_mask, new_deltas_mask_2)
        norms_windows = torch.sqrt(torch.sum((delta_curr * mask_2) ** 2, axis=(2, 3), keepdims=True))

        ### create the updates
        new_deltas = torch.ones([x_curr.shape[0], c, s, s]).to(device)
        new_deltas = new_deltas * meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
        new_deltas *= torch.tensor(np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])).to(device)
        old_deltas = delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
        new_deltas += old_deltas
        new_deltas = new_deltas / torch.sqrt(torch.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
                torch.maximum(self.eps ** 2 - curr_norms_image ** 2, torch.tensor(0)) / c + norms_windows ** 2) ** 0.5
        delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
        delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0  # update window_1

        x_new = x_curr + delta_curr / torch.sqrt(torch.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True)) * self.eps
        x_new = torch.clip(x_new, min_val, max_val)
        curr_norms_image = torch.sqrt(torch.sum((x_new - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
        if curr_norms_image > self.eps * 1.1:
            print(f'eps is {self.eps} but norm is {curr_norms_image}')
        x_hat = self.project(x_new)
        return x_hat

    def linf_mutation(self, x_hat):
        p = self.p_selection(self.p_init, self.queries, self.n_gen * self.pop_size)
        c = x_hat[0].shape[1]
        h = x_hat[0].shape[2]
        w = x_hat[0].shape[3]
        n_features = c * h * w
        s = int(round(np.sqrt(p * n_features / c)))
        s = min(max(s, 1), h - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        x_curr_window = x_hat[0][:, :, center_h:center_h + s, center_w:center_w + s]
        for i in range(c):
            x_curr_window[:, i] += np.random.choice([-2 * self.eps, 2 * self.eps]) * torch.ones(
                x_curr_window[:, i].shape).to(device)

        x_hat[0][:, :, center_h:center_h + s, center_w: center_w + s] = x_curr_window
        x_hat = self.project(x_hat[0])
        return x_hat

    def fitness(self, cur_pop):
        for i in range(len(cur_pop)):
            x_hat, fitness = cur_pop[i]
            x_hat_l2 = F.mse_loss(x_hat, self.x)
            def_x_hat = x_hat.clone()
            if self.defense:
                def_x_hat = self.defense(def_x_hat.cpu().numpy())[0]
                def_x_hat = torch.tensor(def_x_hat).to(device)
            n_x_hat = normalize(self.dataset, def_x_hat)
            probs = F.softmax(self.model(n_x_hat), dim=1).squeeze()
            objective = probs[self.y] - max(x for i, x, in enumerate(probs) if not i == self.y)
            cur_pop[i] = [x_hat, objective + x_hat_l2]

    def get_label(self, x_hat):
        def_x_hat = x_hat.clone()
        if self.defense:
            def_x_hat = self.defense(def_x_hat.cpu().numpy())[0]
            def_x_hat = torch.tensor(def_x_hat).to(device)
        n_x_hat = normalize(self.dataset, def_x_hat)
        return torch.argmax(F.softmax(self.model(n_x_hat), dim=1))

    def termination_condition(self, cur_pop, gen):
        if gen == self.n_gen:
            return True
        for [x_hat, _] in cur_pop:
            y_hat = self.get_label(x_hat)
            self.queries += 1
            if y_hat != self.y:
                self.best_x_hat = x_hat
                return True
        return False

    def project(self, x_hat):
        projected_x_hat = torch.clip(x_hat, self.min_ball, self.max_ball)
        return projected_x_hat

    def pop_init(self):
        if self.norm == 'l2':
            init_func = self.l2_init
        elif self.norm == 'linf':
            init_func = self.vertical_mutation
        else:
            raise ValueError(f'Unrecognized norm: {self.norm}')

        cur_pop = []
        for i in range(self.pop_size):
            x_hat = self.x.clone()
            x_hat = init_func(x_hat)
            cur_pop.append([x_hat, np.inf])
        return cur_pop

    def l2_init(self, x_hat):
        c, h, w = x_hat.shape[1:]
        assert x_hat.shape[0] == 1

        ### initialization
        delta_init = torch.zeros(x_hat.shape).to(device)
        s = h // 5  # s is initial square side for bumps
        sp_init = (h - s * 5) // 2
        center_h = sp_init + 0
        for counter in range(h // s):
            center_w = sp_init + 0
            for counter2 in range(w // s):
                delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += \
                    meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s]) * \
                    torch.from_numpy(np.random.choice([-1, 1], size=[x_hat.shape[0], c, 1, 1])).to(device)
                center_w += s
            center_h += s

        return torch.clip(
            x_hat + delta_init / torch.sqrt(torch.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * self.eps,
            0, 1
        )

    def vertical_mutation(self, x_hat):
        size = np.asarray(self.x.shape)
        size[2] = 1
        x_hat = x_hat + self.eps * torch.tensor(np.random.choice([-1, 1], size=size)).to(device)
        x_hat = self.project(x_hat)
        return x_hat

    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)

        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it <= 10000:
            p = p_init / 512
        else:
            p = p_init

        return p
