import numpy as np
import argparse
import torch
import random

from utils import get_model, correctly_classified, print_initialize, print_success
from data.datasets_loader import load_dataset
from attacks.square_attack import square_attack
from attack import EvoAttack
from utils import compute_accuracy

MODEL_PATH = './models/state_dicts'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models_names = ['custom', 'inception_v3', 'resnet50', 'vgg16_bn', 'vit_l_16']
datasets_names = ['mnist', 'imagenet', 'cifar10']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs Evolutionary Adversarial Attacks on various Deep Learning models")
    parser.add_argument("--norm", "-n", choices=['l2', 'linf'], default='linf', help="Use l_2 or l_inf norm")
    parser.add_argument("--model", "-m", choices=models_names, default='custom',
                        help="Run only specific model")
    parser.add_argument("--dataset", "-da", choices=datasets_names, default='cifar10',
                        help="Run only specific dataset")
    parser.add_argument("--eps", "-ep", type=float, default=0.1,
                        help="Constrained optimization problem - epsilon")
    parser.add_argument("--pop", "-pop", type=int, default=70,
                        help="Population size")
    parser.add_argument("--gen", "-g", type=int, default=60,
                        help="Number of generations")
    parser.add_argument("--images", "-i", type=int, default=201,
                        help="Maximal number of images from dataset to process")
    parser.add_argument("--tournament", "-t", type=int, default=25,
                        help="Tournament selection")
    parser.add_argument("--path", "-ip", default='/cs_storage/public_datasets/ImageNet',
                        help="ImageNet dataset path")
    parser.add_argument('--seed', type=int, default=None, help="Randomization seed; 'none' means to not set the seed")
    args = parser.parse_args()

    if args.seed is not None:
        print(f"Setting seed to {args.seed}")
        np.random.seed(args.seed)
        random.seed(args.seed)

    n_images = args.images
    dataset = args.dataset
    model = args.model
    norm = args.norm
    tournament = args.tournament
    eps = args.eps
    pop_size = args.pop
    n_gen = args.gen
    imagenet_path = args.path
    n_iter = n_gen * pop_size

    (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(dataset, imagenet_path)
    init_model = get_model(model, dataset, MODEL_PATH)
    # compute_accuracy(dataset, init_model, x_test, y_test, min_pixel_value, max_pixel_value, to_normalize=True)
    count = 0
    success_count = 0
    evo_queries = []
    images_indices = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = x.unsqueeze(dim=0).to(device)
        y = y.to(device)

        if count == n_images:
            break

        if correctly_classified(dataset, init_model, x, y) and count < n_images:
            count += 1
            print_initialize(dataset, init_model, x, y, count, n_images)
            images_indices.append(i)
            result = EvoAttack(dataset=dataset, model=init_model, x=x, y=y, eps=eps, n_gen=n_gen,
                               pop_size=pop_size, tournament=tournament, norm=norm, count=count).generate()

            if result['x_hat'] is not None:
                success_count += 1
                adv = result['x_hat'].cpu().numpy()
                if success_count == 1:
                    evo_x_test_adv = adv
                else:
                    evo_x_test_adv = np.concatenate((adv, evo_x_test_adv), axis=0)
                print_success(dataset, init_model, result, y)
            else:
                print('Evolution failed!')
            evo_queries.append(result['queries'])

    print('\n########################################')
    x_test, y_test = x_test[images_indices], y_test[images_indices]
    square_queries, square_adv = [], None
    for i in range(len(x_test)):
        min_ball = torch.tile(torch.maximum(x_test[[i]] - eps, min_pixel_value), (1, 1))
        max_ball = torch.tile(torch.minimum(x_test[[i]] + eps, max_pixel_value), (1, 1))

        square_adv, square_n_queries = square_attack(dataset, init_model, min_ball, max_ball, x_test[[i]], i,
                                                     square_adv, n_iter, eps, norm)
        square_queries.append(square_n_queries)

    square_accuracy = compute_accuracy(dataset, init_model, square_adv, y_test, min_pixel_value, max_pixel_value,
                                       to_tensor=True, to_normalize=True)

    print()
    print('########################################')
    print(f'Summary:')
    print(f'\tDataset: {dataset}')
    print(f'\tModel: {model}')
    print(f'\tNorm: {norm}')
    print(f'\tTournament: {tournament}')
    print(f'\tMetric: linf, epsilon: {eps:.4f}')
    print(f'\tSquare:')
    print(f'\t\tSquare - attack success rate: {(1 - square_accuracy) * 100:.4f}%')
    print(f'\t\tSquare - queries: {square_queries}')
    print(f'\t\tSquare - queries (median): {int(np.median(square_queries))}')
    print(f'\tEvo:')
    print(f'\t\tEvo - attack success rate: {(success_count / n_images) * 100:.4f}%')
    print(f'\t\tEvo - queries: {evo_queries}')
    print(f'\t\tEvo - queries (median): {int(np.median(evo_queries))}')
    print('########################################')
