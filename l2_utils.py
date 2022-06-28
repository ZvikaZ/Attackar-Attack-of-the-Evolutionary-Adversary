import random
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# TODO work only with torch instead of numpy?

def get_perturbation(height):
    delta = np.zeros([height, height])
    gaussian_perturbation = np.zeros([height // 2, height])

    x_c = height // 4
    y_c = height // 2

    for i_y in range(y_c):
        gaussian_perturbation[
        max(x_c, 0): min(x_c + (2 * i_y + 1), height // 2),
        max(0, y_c): min(y_c + (2 * i_y + 1), height),
        ] += 1.0 / ((i_y + 1) ** 2)
        x_c -= 1
        y_c -= 1

    gaussian_perturbation /= np.sqrt(np.sum(gaussian_perturbation ** 2))

    delta[: height // 2] = gaussian_perturbation
    delta[height // 2: height // 2 + gaussian_perturbation.shape[0]] = -gaussian_perturbation

    delta /= np.sqrt(np.sum(delta ** 2))

    if random.random() > 0.5:
        delta = np.transpose(delta)

    if random.random() > 0.5:
        delta = -delta

    return delta


# TODO remove the following 2 methods

def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1

    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
        max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def meta_pseudo_gaussian_pert(s):
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5: delta = np.transpose(delta)

    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return torch.from_numpy(delta).to(device)
