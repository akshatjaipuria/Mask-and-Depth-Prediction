import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self, drop=0):
        super(Net, self).__init__()
        self.drop = drop

        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False, groups=32),
            nn.Conv2d(32, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(128, 1, 3, stride=1, padding=1, bias=False),
            # nn.ReLU()
        )

    def forward(self, input1, input2):
        x1 = input1
        x2 = input2

        x1 = self.convblock2(self.convblock1(x1))
        x2 = self.convblock2(self.convblock1(x2))

        x_1 = (x1 + x2)
        # x = self.convblock3(x_1)
        # x = self.convblock4(x)
        # x = nn.ReLU()(x + x_1)
        # x = torch.cat([x1, x2], dim=1)

        x = self.convblock6(self.convblock5(x))

        return x
