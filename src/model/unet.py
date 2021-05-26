import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------- #
#                                  Unet model                                       #
#    This piece of code was taken from https://github.com/milesial/Pytorch-UNet     #
# --------------------------------------------------------------------------------- #


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm="BN", num_groups=4, norm_momentum=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm == "BN":
            self.bn1 = nn.BatchNorm2d(mid_channels, momentum=norm_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=norm_momentum)
        elif norm == "GN" and num_groups != 0:
            print("Using group norm!")
            self.bn1 = nn.GroupNorm(num_groups, mid_channels)
            self.bn2 = nn.GroupNorm(num_groups, out_channels)
        else:
            raise RuntimeError("wrong norm option: {}".format(norm))
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            self.bn1,
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            self.bn2,
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm="BN", num_groups=4):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm=norm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm="BN", num_groups=4):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm, num_groups=num_groups)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm, num_groups=num_groups)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, out_relu=False):
        super(OutConv, self).__init__()
        self.out_relu = out_relu
        if self.out_relu:
            self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        if self.out_relu:
            x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, filters=[32, 64, 128, 256, 512], bilinear=True, out_relu=False, norm="BN", num_groups=4,
        sparse_sinogram_net=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sparse_sinogram_net = sparse_sinogram_net
        if sparse_sinogram_net:
            self.up = nn.Upsample(scale_factor=(4, 1), mode='bicubic', align_corners=False)

        self.inc = DoubleConv(n_channels, filters[0], norm=norm, num_groups=num_groups)
        self.down1 = Down(filters[0], filters[1], norm=norm, num_groups=num_groups)
        self.down2 = Down(filters[1], filters[2], norm=norm, num_groups=num_groups)
        self.down3 = Down(filters[2], filters[3], norm=norm, num_groups=num_groups)
        factor = 2 if bilinear else 1
        self.down4 = Down(filters[3], filters[4] // factor, norm=norm, num_groups=num_groups)
        self.up1 = Up(filters[4], filters[3] // factor, bilinear, norm=norm, num_groups=num_groups)
        self.up2 = Up(filters[3], filters[2] // factor, bilinear, norm=norm, num_groups=num_groups)
        self.up3 = Up(filters[2], filters[1] // factor, bilinear, norm=norm, num_groups=num_groups)
        self.up4 = Up(filters[1], filters[0], bilinear, norm=norm, num_groups=num_groups)
        self.outc = OutConv(filters[0], n_classes, out_relu=out_relu)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.sparse_sinogram_net:
            x = self.up(x)
            res = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.sparse_sinogram_net:
            logits = self.outc(x) + res
        else:
            logits = self.outc(x)
        logits = self.relu(logits)
        return logits
