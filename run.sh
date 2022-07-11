#!/bin/bash
#SBATCH --job-name attackar_ng
#SBATCH --partition main
#SBATCH --qos normal
##SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
##SBATCH --mem=4G              ### amount of RAM memory

which python
hostname
# python main.py --model inception_v3 --dataset imagenet --eps 5 --norm l2    # --images 1
python -u main.py --model inception_v3 --dataset imagenet --eps 5 --norm l2 --images 200 --gen 30 --pop 6

#python main.py --model inception_v3 --dataset imagenet --eps 0.025
#python main.py --model custom --dataset mnist --eps 0.225
