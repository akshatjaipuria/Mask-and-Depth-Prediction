import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary
import torch
from torch import nn
import numpy as np


def show_image(inp, n_row=8, title=None, mean=None, std=None):
    """Imshow for Tensor."""
    inp = torchvision.utils.make_grid(inp.detach().cpu(), n_row)
    inp = inp.numpy().transpose((1, 2, 0))
    if mean:
        mean = np.array(mean)
        std = np.array(std)
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def model_summary(net, size):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = net.to(device)
    print(summary(model, input_size=size))
    return device


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        # probs = F.sigmoid(logits)
        num = targets.size(0)  # Number of batches

        score = dice_coeff(logits, targets)
        score = 1 - score.sum() / num
        return score


def to_numpy(tensor):
    """tensor -> (B,C,H,W) to numpy array -> (B,H,W,C)"""
    return np.transpose(tensor.clone().numpy(), (0, 2, 3, 1))
