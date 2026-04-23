import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(image, ax=None, normalize=False):
    if ax is None:
        fig, ax = plt.subplots()

    # Jos tensor on GPU:lla, siirretään CPU:lle
    if isinstance(image, torch.Tensor):
        image = image.cpu().clone().detach()

    # Muoto: C x H x W  -> H x W x C
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        image = image - image.min()
        if image.max() != 0:
            image = image / image.max()

    ax.imshow(image)
    ax.axis("off")

    return ax