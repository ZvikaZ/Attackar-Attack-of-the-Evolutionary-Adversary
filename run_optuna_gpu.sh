sbatch --wrap="python -u run_optuna.py" --gpus=1 --job-name="optuna"
