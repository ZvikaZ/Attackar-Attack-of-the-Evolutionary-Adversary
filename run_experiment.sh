#!/bin/bash
mkdir -p results

ds=imagenet
model=inception_v3
eps=5
images=20

for gen in 1 10 20 40 60 100
do
  for pop in 1 10 20 40 60 100
  do
    sbatch --gpus=1 --wrap="python main.py --model $model --dataset $ds --eps $eps --images $images --norm l2 --gen $gen --pop $pop > results/l2_g_${gen}_p_${pop}.log"
  done
done
