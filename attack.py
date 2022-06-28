import torch.nn.functional as F
from operator import itemgetter
import numpy as np
import random
import torch
import math

from utils import normalize, plt_torch_image

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

    def l2_mutation(self, x_hat_orig):
        channels_first = True  # TODO make it configurable, or remove all those 'if's
        min_val, max_val = 0, 1

        if channels_first:
            channels = self.x.shape[1]
            height = self.x.shape[2]
            width = self.x.shape[3]
        else:
            height = self.x.shape[1]
            width = self.x.shape[2]
            channels = self.x.shape[3]

        # TODO move all NP to PyTorch
        # currently we transfer all torch tensors to np arrays
        x_hat = x_hat_orig[0].cpu().detach().numpy()
        x_orig = self.x.cpu().detach().numpy()

        percentage_of_elements = self.p_selection(self.p_init, self.queries, self.n_gen * self.pop_size)

        delta_x_hat_init = x_hat - x_orig

        height_tile = max(int(round(math.sqrt(percentage_of_elements * height * width))), 3)

        if height_tile % 2 == 0:
            height_tile += 1
        height_tile_2 = height_tile

        height_start = np.random.randint(0, height - height_tile)
        width_start = np.random.randint(0, width - height_tile)

        new_deltas_mask = np.zeros(x_orig.shape)
        if channels_first:
            new_deltas_mask[
            :, :, height_start: height_start + height_tile, width_start: width_start + height_tile
            ] = 1.0
            w_1_norm = np.sqrt(
                np.sum(
                    delta_x_hat_init[
                    :,
                    :,
                    height_start: height_start + height_tile,
                    width_start: width_start + height_tile,
                    ]
                    ** 2,
                    axis=(2, 3),
                    keepdims=True,
                )
            )
        else:
            new_deltas_mask[
            :, height_start: height_start + height_tile, width_start: width_start + height_tile, :
            ] = 1.0
            w_1_norm = np.sqrt(
                np.sum(
                    delta_x_hat_init[
                    :,
                    height_start: height_start + height_tile,
                    width_start: width_start + height_tile,
                    :,
                    ]
                    ** 2,
                    axis=(1, 2),
                    keepdims=True,
                )
            )

        height_2_start = np.random.randint(0, height - height_tile_2)
        width_2_start = np.random.randint(0, width - height_tile_2)

        new_deltas_mask_2 = np.zeros(x_orig.shape)
        if channels_first:
            new_deltas_mask_2[
            :,
            :,
            height_2_start: height_2_start + height_tile_2,
            width_2_start: width_2_start + height_tile_2,
            ] = 1.0
        else:
            new_deltas_mask_2[
            :,
            height_2_start: height_2_start + height_tile_2,
            width_2_start: width_2_start + height_tile_2,
            :,
            ] = 1.0

        norms_x_hat = np.sqrt(np.sum((x_hat - x_orig) ** 2, axis=(1, 2, 3), keepdims=True))
        w_norm = np.sqrt(
            np.sum(
                (delta_x_hat_init * np.maximum(new_deltas_mask, new_deltas_mask_2)) ** 2,
                axis=(1, 2, 3),
                keepdims=True,
            )
        )

        if channels_first:
            new_deltas_size = [x_orig.shape[0], channels, height_tile, height_tile]
            random_choice_size = [x_orig.shape[0], channels, 1, 1]
            perturbation_size = (1, 1, height_tile, height_tile)
        else:
            new_deltas_size = [x_orig.shape[0], height_tile, height_tile, channels]
            random_choice_size = [x_orig.shape[0], 1, 1, channels]
            perturbation_size = (1, height_tile, height_tile, 1)

        delta_new = (
                np.ones(new_deltas_size)
                * self.get_l2_perturbation(height_tile).reshape(perturbation_size)
                * np.random.choice([-1, 1], size=random_choice_size)
        )

        if channels_first:
            delta_new += delta_x_hat_init[
                         :, :, height_start: height_start + height_tile, width_start: width_start + height_tile
                         ] / (np.maximum(1e-9, w_1_norm))
        else:
            delta_new += delta_x_hat_init[
                         :, height_start: height_start + height_tile, width_start: width_start + height_tile, :
                         ] / (np.maximum(1e-9, w_1_norm))

        diff_norm = (self.eps * np.ones(delta_new.shape)) ** 2 - norms_x_hat ** 2
        diff_norm[diff_norm < 0.0] = 0.0

        if channels_first:
            delta_new /= np.sqrt(np.sum(delta_new ** 2, axis=(2, 3), keepdims=True)) * np.sqrt(
                diff_norm / channels + w_norm ** 2
            )
            delta_x_hat_init[
            :,
            :,
            height_2_start: height_2_start + height_tile_2,
            width_2_start: width_2_start + height_tile_2,
            ] = 0.0
            delta_x_hat_init[
            :, :, height_start: height_start + height_tile, width_start: width_start + height_tile
            ] = delta_new
        else:
            delta_new /= np.sqrt(np.sum(delta_new ** 2, axis=(1, 2), keepdims=True)) * np.sqrt(
                diff_norm / channels + w_norm ** 2
            )
            delta_x_hat_init[
            :,
            height_2_start: height_2_start + height_tile_2,
            width_2_start: width_2_start + height_tile_2,
            :,
            ] = 0.0
            delta_x_hat_init[
            :, height_start: height_start + height_tile, width_start: width_start + height_tile, :
            ] = delta_new

        x_hat_new = np.clip(
            x_orig
            + self.eps
            * delta_x_hat_init
            / np.sqrt(np.sum(delta_x_hat_init ** 2, axis=(1, 2, 3), keepdims=True)),
            min_val,
            max_val,
        )
        return torch.from_numpy(x_hat_new).to(device)

    def get_l2_perturbation(self, height):
        delta = np.zeros([height, height])
        gaussian_perturbation = np.zeros([height // 2, height])

        x_c = height // 4
        y_c = height // 2

        for i_y in range(y_c):
            gaussian_perturbation[
            max(x_c, 0): min(x_c + (2 * i_y + 1), height // 2),
            max(0, y_c): min(y_c + (2 * i_y + 1), height),
            ] += 1.0 / ((i_y + 1) ** 2)
            x_c -= 1
            y_c -= 1

        gaussian_perturbation /= np.sqrt(np.sum(gaussian_perturbation ** 2))

        delta[: height // 2] = gaussian_perturbation
        delta[height // 2: height // 2 + gaussian_perturbation.shape[0]] = -gaussian_perturbation

        delta /= np.sqrt(np.sum(delta ** 2))

        if random.random() > 0.5:
            delta = np.transpose(delta)

        if random.random() > 0.5:
            delta = -delta

        return delta

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
        channels_first = True  # TODO make it configurable, or remove all those 'if's
        min_val, max_val = 0, 1

        if channels_first:
            channels = self.x.shape[1]
            height = self.x.shape[2]
            width = self.x.shape[3]
        else:
            height = self.x.shape[1]
            width = self.x.shape[2]
            channels = self.x.shape[3]

        n_tiles = 5
        height_tile = height // n_tiles

        delta_init = np.zeros(x_hat.shape, dtype=np.float32)

        height_start = 0
        for _ in range(n_tiles):
            width_start = 0
            for _ in range(n_tiles):
                if channels_first:
                    perturbation_size = (1, 1, height_tile, height_tile)
                    random_size = (channels, 1, 1)
                else:
                    perturbation_size = (1, height_tile, height_tile, 1)
                    random_size = (1, 1, channels)

                perturbation = self.get_l2_perturbation(height_tile).reshape(perturbation_size) * np.random.choice(
                    [-1, 1], size=random_size
                )

                if channels_first:
                    delta_init[
                    :, :, height_start: height_start + height_tile, width_start: width_start + height_tile
                    ] += perturbation
                else:
                    delta_init[
                    :, height_start: height_start + height_tile, width_start: width_start + height_tile, :
                    ] += perturbation
                width_start += height_tile
            height_start += height_tile

        x_hat_new = np.clip(
            x_hat.cpu().detach().numpy() + delta_init / np.sqrt(
                np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * self.eps,
            min_val,
            max_val,
        )
        return torch.from_numpy(x_hat_new).to(device)

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
