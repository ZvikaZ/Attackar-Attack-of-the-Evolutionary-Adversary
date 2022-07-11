mamba create -n attackar_ng python=3.8.12 numpy=1.21 tensorflow=2.8 matplotlib pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch
mamba activate attackar_ng
pip install adversarial-robustness-toolbox==1.10
pip install git+https://github.com/RobustBench/robustbench.git@f8690ed4fb4fb2a04439a336de6777742d34b897
pip install git+https://github.com/ZvikaZ/pslurm
pip install sorcery

# to run optuna experiments:
mamba install -c conda-forge optuna
mamba install -c anaconda joblib
#mamba install -c plotly plotly
