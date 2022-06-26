from art.estimators.classification import PyTorchClassifier
from robustbench.utils import load_model
from torchvision.utils import save_image
from torchvision import transforms
import torchvision.models
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

from models.inception import inception_v3
from models.resnet import resnet50
from models.vgg import vgg16_bn

print("Checking for gpu...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")


def compute_accuracy(dataset, init_model, x_test, y_test, min_pixel_value, max_pixel_value, to_tensor=False,
                     to_normalize=False):
    classifier = PyTorchClassifier(
        model=init_model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=nn.CrossEntropyLoss(),
        input_shape=x_test[0].shape,
        nb_classes=10,
    )
    if to_tensor:
        x_test = torch.from_numpy(x_test)
    if to_normalize:
        x_test = normalize(dataset, x_test)
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.array(y_test)) / len(y_test)
    print("Accuracy: {}%".format(accuracy * 100))
    return accuracy


def correctly_classified(dataset, model, x, y):
    softmax = nn.Softmax(dim=1)
    x = normalize(dataset, x)
    return y == torch.argmax(softmax(model(x)))


def get_model(model_name, dataset, path):
    if model_name == 'custom':
        model = torch.load(Path(path) / f'{dataset}_model.pth', map_location=torch.device(device)).eval()
    elif dataset == 'cifar10':
        if model_name == 'gowal':
            model = load_model(model_name='Gowal2020Uncovering_70_16', dataset='cifar10', threat_model='Linf').to(
                device)
        elif model_name == 'rebuffi':
            model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10',
                               threat_mode='Linf').to(device)
        else:
            model = globals()[model_name](pretrained=True).to(device).eval()
    elif dataset == 'imagenet':
        if model_name == 'vgg16_bn':
            model = torchvision.models.vgg16_bn(pretrained=True).to(device).eval()
        elif model_name == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True).to(device).eval()
        elif model_name == 'inception_v3':
            model = torchvision.models.inception_v3(pretrained=True).to(device).eval()
        elif model_name == 'salman':
            model = load_model(model_name='Salman2020Do_50_2', dataset='imagenet', threat_model='Linf').to(device)
    else:
        raise Exception('No such dataset!')
    return model


def normalize(dataset, images):
    if dataset == 'cifar10':
        norm_images = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))(images)
    elif dataset == 'mnist':
        norm_images = transforms.Normalize((0.5,), (0.5,))(images)
    elif dataset == 'imagenet':
        norm_images = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(images)
    return norm_images


def get_normalization(dataset):
    if dataset == 'cifar10':
        values = ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    elif dataset == 'mnist':
        values = ((0.5,), (0.5,))
    elif dataset == 'imagenet':
        values = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return values


def inv_normalize(dataset):
    if dataset == 'cifar10':
        return transforms.Normalize(
            mean=[-0.4914 / 0.2471, -0.4822 / 0.2435, -0.4465 / 0.2616],
            std=[1 / 0.2471, 1 / 0.2435, 1 / 0.2616]
        )
    elif dataset == 'mnist':
        return transforms.Normalize(mean=[-0.5 / 0.5], std=[1 / 0.5])

    elif dataset == 'imagenet':
        return transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
    else:
        return 'WTF'


def inv_normalize_and_save(img, best_individual, not_best_individual):
    orig = img
    if (not_best_individual != None and best_individual != None):
        good_attack = best_individual
        bad_attack = not_best_individual
        save_image(good_attack, 'good.png')
        save_image(bad_attack, 'bad.png')
        save_image(orig, 'orig.png')


def print_initialize(dataset, model, img, label, count, n_images):
    normalized_img = normalize(dataset, img)
    print("################################")
    print(f'Image {count} of {n_images}')
    print(f'Correct class: {label}')
    print(f'Initial class prediction: {model(normalized_img).argmax(dim=1).item()}')
    print(f'Initial probability: {F.softmax(model(normalized_img), dim=1).max():.4f}')


def print_success(dataset, model, result, label):
    best_individual = result['x_hat'].clone().detach().to(device)
    normalized_best_inv = normalize(dataset, best_individual)
    print("-------------------------------")
    print(f'Evolution succeeded!')
    print(f'Correct class: {label}')
    print(f'Current prediction: {model(normalized_best_inv).argmax(dim=1).item()}')
    print(
        f'Current probability (orig class): {F.softmax(model(normalized_best_inv), dim=1)[0][label].item():.4f}')
    print(f'Number of queries: {result["queries"]}')
    print(f'Number of generations: {result["gen"]}')
