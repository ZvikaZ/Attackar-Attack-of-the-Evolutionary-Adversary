#!/bin/bash
mkdir -p results
for ds in imagenet cifar10
do    
    for model in inception_v3 resnet50 vgg16_bn
    do
        for eps in 0.1 0.075 0.05 0.025
        do
            sbatch --gpus=1 --wrap="python main.py --model $model --dataset $ds --eps $eps > results/${model}_${ds}_${eps}.log"
        done
    done
done

ds=mnist
model=custom
eps=0.225
sbatch --gpus=1 --wrap="python main.py --model $model --dataset $ds --eps $eps > results/${model}_${ds}_${eps}.log"
eps=0.3
sbatch --gpus=1 --wrap="python main.py --model $model --dataset $ds --eps $eps > results/${model}_${ds}_${eps}.log"
