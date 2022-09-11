import numpy as np
import pandas as pd
import argparse
import torch
import random
import warnings

import pslurm
from func_slurm import FuncSlurm

from evo_attack import EvoAttack
from utils import get_model, correctly_classified, print_initialize, print_success, compute_accuracy, get_median_index
from data.datasets_loader import load_dataset
from attacks.square_attack import square_attack

MODEL_PATH = './models/state_dicts'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models_names = ['custom', 'inception_v3', 'resnet50', 'vgg16_bn', 'vit_l_16']
datasets_names = ['mnist', 'imagenet', 'cifar10']


def generate_evo_attack(dataset, model, x, y, eps, n_gen, pop_size, tournament, norm, i):
    # needed for FuncSlurm
    return EvoAttack(dataset=dataset, model=model, x=x, y=y, n_gen=n_gen, pop_size=pop_size, eps=eps,
                     tournament=tournament, norm=norm, i=i).generate()


# TODO use_slurm
def attack(n_images, dataset, model, norm, tournament, eps, pop_size, n_gen, imagenet_path, n_iter, repeats, use_slurm):
    (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(dataset, imagenet_path)
    init_model = get_model(model, dataset, MODEL_PATH)
    # compute_accuracy(dataset, init_model, x_test, y_test, min_pixel_value, max_pixel_value, to_normalize=True)
    success_count = 0
    evo_queries = []
    images_indices = []
    jobs = []
    # TODO take more images if not all are used (not correctly_classified(..))
    for i, (x, y) in enumerate(zip(x_test[:n_images], y_test[:n_images])):
        jobs.append(FuncSlurm(evo_single_image, dataset, eps, i, init_model, n_gen, n_images, norm, pop_size, repeats,
                              tournament, x, y))
    for job in jobs:
        result = job.get_result()
        if result:
            images_indices.append(i)
            if result['success']:
                success_count += 1
                # TODO is it used? commented because seems redundant
                # adv = result['x_hat'].cpu().numpy()
                # if success_count == 1:
                #     evo_x_test_adv = adv
                # else:
                #     evo_x_test_adv = np.concatenate((adv, evo_x_test_adv), axis=0)
                print_success(dataset, init_model, result, y)
            else:
                print('Evolution failed!')
            evo_queries.append(result['queries'])  # TODO maybe measure only successfull queries?

    evo_asr = success_count / len(images_indices) * 100
    evo_queries_median = np.median(evo_queries)

    print('\n########################################')
    x_test, y_test = x_test[images_indices], y_test[images_indices]
    square_queries, square_adv = [], None
    for i in range(len(x_test)):
        print(f'Running square attack on image #{i}')
        min_ball = torch.tile(torch.maximum(x_test[[i]] - eps, min_pixel_value), (1, 1))
        max_ball = torch.tile(torch.minimum(x_test[[i]] + eps, max_pixel_value), (1, 1))

        jobs = []
        for k in range(repeats):
            jobs.append(FuncSlurm(
                square_attack, dataset, init_model, min_ball, max_ball, x_test[[i]], i, square_adv, n_iter, eps, norm))
        results = []
        for job in jobs:
            square_adv, square_n_queries = job.get_result()
            results.append({'adv': square_adv, 'queries': square_n_queries})
        square_adv, square_n_queries = avarage_square_results(results)

        square_queries.append(square_n_queries)
    square_accuracy = compute_accuracy(dataset, init_model, square_adv, y_test, min_pixel_value, max_pixel_value,
                                       to_tensor=True, to_normalize=True)
    square_asr = (1 - square_accuracy) * 100
    square_queries_median = np.median(square_queries)

    delta_queries = evo_queries_median - square_queries_median
    delta_asr = evo_asr - square_asr

    print()
    print('########################################')
    print(f'Summary of run:')
    print(f'\tDataset: {dataset}')
    print(f'\tModel: {model}')
    print(f'\tNorm: {norm}')
    print(f'\tTournament: {tournament}')
    print(f'\tMetric: linf, epsilon: {eps:.4f}')
    print(f'\tSquare:')
    print(f'\t\tSquare - attack success rate: {square_asr:.4f}%')
    print(f'\t\tSquare - queries: {square_queries}')
    print(f'\t\tSquare - queries (median): {int(square_queries_median)}')
    print(f'\tEvo:')
    print(f'\t\tEvo - attack success rate: {evo_asr:.4f}%')
    print(f'\t\tEvo - queries: {evo_queries}')
    try:
        print(f'\t\tEvo - queries (median): {int(np.median(evo_queries))}')
    except ValueError:
        print(f'\t\tEvo - queries (median): NaN')
    print(f'\tSummary:')
    print(f'\t\tDelta - attack success rate: {delta_asr:.4f}%')
    print(f'\t\tDelta - queries: {delta_queries}')
    print('########################################')

    return {
        'square_asr': square_asr,
        'square_queries_median': square_queries_median,
        'evo_asr': evo_asr,
        'evo_queries_median': evo_queries_median,
        'delta_queries': delta_queries,
        'delta_asr': delta_asr,
    }


def evo_single_image(dataset, eps, i, init_model, n_gen, n_images, norm, pop_size, repeats, tournament, x, y):
    x = x.unsqueeze(dim=0).to(device)
    y = y.to(device)
    if correctly_classified(dataset, init_model, x, y):
        print_initialize(dataset, init_model, x, y, i, n_images)
        jobs = []
        for i in range(repeats):
            jobs.append(
                FuncSlurm(generate_evo_attack, dataset=dataset, model=init_model, x=x, y=y, eps=eps, n_gen=n_gen,
                          pop_size=pop_size, tournament=tournament, norm=norm, i=i))
        results = []
        for job in jobs:
            try:
                results.append(job.get_result())
            except Exception as e:
                warnings.warn('Attackar: FunSlurm got exception: ' + str(e))
        return avarage_evo_results(results)
    else:
        return None


def avarage_square_results(results):
    df = pd.DataFrame(results)
    median_queries_row = list(df.loc[[get_median_index(df)['queries']]].T.to_dict().values())[0]
    return median_queries_row['adv'], median_queries_row['queries']


def avarage_evo_results(results):
    if not results:
        print(results)  # TODO del
        print('No results')  # TODO del
        return None
    result = {}
    df = pd.DataFrame(results)
    if df.count()['x_hat'] > len(df) // 2:
        # more than half returned a valid x_hat
        result['success'] = True
        # keep only good results
        df = df.dropna()
    else:
        result['success'] = False
    median_queries_row = list(df.loc[[get_median_index(df)['queries']]].T.to_dict().values())[0]
    result.update(median_queries_row)
    return result


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
    parser.add_argument('--slurm', action='store_true', help='Use Slurm HPC (default)')
    parser.add_argument('--no-slurm', dest='slurm', action='store_false', help="Don't use Slurm HPC")
    parser.set_defaults(slurm=True)
    parser.add_argument('--repeats', type=int, default=10,
                        help="How many item to repeat each image (summary results are averaged)")
    args = parser.parse_args()

    if args.seed is not None:
        print(f"Setting seed to {args.seed}")
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.slurm:
        assert pslurm.is_slurm_installed()

    attack(n_images=args.images,
           dataset=args.dataset,
           model=args.model,
           norm=args.norm,
           tournament=args.tournament,
           eps=args.eps,
           pop_size=args.pop,
           n_gen=args.gen,
           imagenet_path=args.path,
           n_iter=args.gen * args.pop,
           repeats=args.repeats,
           use_slurm=args.slurm)
