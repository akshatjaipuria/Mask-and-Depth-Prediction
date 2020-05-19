import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self, drop=0):
        super(Net, self).__init__()
        self.drop = drop

        self.convblock1_1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.convblock1_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False, groups=32),
            nn.Conv2d(32, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.convblock2_1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.convblock2_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False, groups=32),
            nn.Conv2d(32, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(128, 1, 3, stride=1, padding=1, bias=False),
        )

    def forward(self, input1, input2):
        x1 = input1
        x2 = input2

        x1 = self.convblock1_2(self.convblock1_1(x1))
        x2 = self.convblock2_2(self.convblock2_1(x2))

        x_1 = torch.cat([x1, x2], dim=1)
        x = self.convblock3(x_1)
        x = self.convblock4(x)
        x = (x + x_1)
        x = self.convblock5(x)
        x = self.convblock6(x)

        return x
