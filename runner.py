import argparse
from pathlib import Path
import torch

from train import load_dataset
from attack import EvoAttack
from models.googlenet import googlenet
from models.densenet import densenet121, densenet169
from models.resnet import resnet18
from models.mobilenetv2 import mobilenet_v2

import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")

def get_model(model_name, dataset):
    if model_name == 'custom':
        return torch.load(Path('models/state_dicts') / f'{dataset}_model.pth', map_location=torch.device(device))
    elif dataset=='cifar10':
        return globals()[model_name](pretrained=True).to(device)
    elif dataset=='imagenet':
        return models.mobilenet_v2(pretrained=True).to(device)
    else:
        raise Exception('No such dataset!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Shouldn't be directly run! use 'main.py' instead")
    parser.add_argument("model_name")
    parser.add_argument("i", type=int)
    parser.add_argument("dataset")
    parser.add_argument("metric")
    args = parser.parse_args()

    model = get_model(args.model_name, args.dataset)
    model.eval()

    train_loader, _ = load_dataset(args.dataset)
    images, labels = next(iter(train_loader))

    img = images[args.i].unsqueeze(dim=0).to(device)
    label = labels[args.i].to(device)

    print("============================")
    print(f"Running attack on {args.model_name} model with metric: {args.metric}, image #{args.i}")
    print("============================")

    EvoAttack(model, img, label, metric=args.metric).evolve()
