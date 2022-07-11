import argparse

import numpy as np
import optuna
import logging
import sys
import matplotlib.pyplot as plt

from main import run_repeats

study_name = "Attacker-params"
storage_name = f"sqlite:///{study_name}.db"


def objective(trial):
    pop_size = trial.suggest_int('pop_size', 1, 80)
    gen = trial.suggest_int('gen', 1, 80)
    try:
        result = run_repeats(n_images=20,
                             repeats=8,
                             dataset='imagenet',
                             model='inception_v3',
                             norm='l2',
                             tournament=25,
                             eps=5,
                             pop_size=pop_size,
                             n_gen=gen,
                             imagenet_path='/cs_storage/public_datasets/ImageNet',
                             n_iter=pop_size * gen,
                             slurm=True)
    except RuntimeError as e:
        print("Optuna trial failed with:")
        print(e)
        return float('nan')
    return result['delta_queries'], result['delta_asr']


def run_experiment():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                                directions=["minimize", "maximize"])

    study.optimize(objective, n_trials=100)

    print('Best trials:')
    for trial in study.best_trials:
        print(trial)

    # these aren't relevant for multi objective:
    # print('Best (queries, asr): ', end='')
    # print(study.best_value)
    # print('Best params: ', end='')
    # print(study.best_params)


def show_results():
    # visualisations from https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                                directions=["minimize", "maximize"])

    print('Best trials:')
    for trial in study.best_trials:
        print(trial)

    optuna.visualization.matplotlib.plot_pareto_front(study, target_names=["queries", "asr"])
    optuna.visualization.matplotlib.plot_param_importances(
        study, target=lambda t: t.values[0], target_name="queries"
    )
    optuna.visualization.matplotlib.plot_param_importances(
        study, target=lambda t: t.values[0], target_name="asr"
    )
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs experiments to find evolutionary hyperparameters")
    parser.add_argument('--results', '-r', action='store_true', help='Show experiments results')
    args = parser.parse_args()

    if args.results:
        show_results()
    else:
        run_experiment()
