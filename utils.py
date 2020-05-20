import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary
import torch
from torch import nn


def show_image(inp, n_row=8, title=None):
    """Imshow for Tensor."""
    inp = torchvision.utils.make_grid(inp.detach().cpu(), n_row)
    inp = inp.numpy().transpose((1, 2, 0))
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     inp = std * inp + mean
    #     inp = np.clip(inp, 0, 1)
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


def dice_coeff(pred, targ):
    smooth = 1.

    pflat = pred.view(-1)
    tflat = targ.view(-1)
    intersection = (pflat * tflat).sum()

    return (2. * intersection + smooth) / (pflat.sum() + tflat.sum() + smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        score = dice_coeff(logits, targets)
        score = 1 - score
        return score


def threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pred, targ, threshold, eps=1e-7):
    pred = threshold(pred, threshold)
    targ = threshold(targ, threshold)

    pflat = pred.view(-1)
    tflat = targ.view(-1)
    intersection = (pflat * tflat).sum()

    return (intersection + eps) / (pflat.sum() + tflat.sum() - intersection + eps)
