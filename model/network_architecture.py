import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OutConv_Depth(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        return self.conv(x)


class Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.up3_depth = Up(128, 64, bilinear)
        self.outc_depth = OutConv_Depth(64, n_classes)

    def forward(self, input1, input2):
        x = torch.cat([input1, input2], dim=1)  # 3 x 224 x 224, 3 x 224 x 224 -> 6 x 224 x 224
        x1 = self.inc(x)     # 6 x 224 x 224   -> 64 x 224 x 224
        x2 = self.down1(x1)  # 64 x 224 x 224  -> 128 x 112 x 112
        x3 = self.down2(x2)  # 128 x 112 x 112 -> 256 x 56 x 56
        x4 = self.down3(x3)  # 256 x 56 x 56   -> 256 x 28 x 28
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up1(x4, x3)   # 256 x 28 x 28, 256 x 56 x 56   -> 128 x 56 x 56
        x = self.up2(x, x2)    # 128 x 56 x 56, 128 x 112 x 112 -> 64 x 112 x 112

        # x_mask = self.up3(x, x1)    # 64 x 112 x 112, 64 x 224 x 224  -> 64 x 224 x 224
        # logits_mask = self.outc(x_mask)  # 64 x 224 x 224, 1/2 x 224 s 224

        x_depth = self.up3_depth(x, x1)
        logits_depth = self.outc_depth(x_depth)

        return logits_depth  # , logits_mask
