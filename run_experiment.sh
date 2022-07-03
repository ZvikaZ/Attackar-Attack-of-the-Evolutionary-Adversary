#!/bin/bash
seed=3

mkdir -p results_$seed

ds=imagenet
model=inception_v3
eps=5
images=100

for gen in 1 2 4 7 10 13 16 19 22 25 28 31 34 37 40
do
  for pop in 1 2 4 7 10 13 16 19 22 25 28 31 34 37 40
  do
    sbatch --gpus=1 --wrap="python main.py --model $model --dataset $ds \
           --eps $eps --images $images --norm l2 --gen $gen --pop $pop --seed $seed > results_$seed/l2_g_${gen}_p_${pop}.log"
  done
done
