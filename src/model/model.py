import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------- #
#                                  Unet model                                       #
#    This piece of code was taken from https://github.com/milesial/Pytorch-UNet     #
# --------------------------------------------------------------------------------- #


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size) 
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mode='bilinear', kernel_size=3):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size=kernel_size)
        elif mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2) # nearest neighbor
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size=kernel_size)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2) # Pas une bonne idée
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size)


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
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, mode='bilinear', channels_depth_number=(64, 128, 256, 512, 1024), kernel_size=3):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        # diviser par 2 les channels. triple conv?
        self.inc = DoubleConv(n_channels, channels_depth_number[0], kernel_size=kernel_size)
        self.down1 = Down(channels_depth_number[0], channels_depth_number[1], kernel_size=kernel_size)
        self.down2 = Down(channels_depth_number[1], channels_depth_number[2], kernel_size=kernel_size)
        self.down3 = Down(channels_depth_number[2], channels_depth_number[3], kernel_size=kernel_size)
        factor = 2 if mode == 'bilinear' else 1
        self.down4 = Down(channels_depth_number[3], channels_depth_number[4] // factor, kernel_size=kernel_size)
        self.up1 = Up(channels_depth_number[4], channels_depth_number[3] // factor, mode, kernel_size=kernel_size)
        self.up2 = Up(channels_depth_number[3], channels_depth_number[2] // factor, mode, kernel_size=kernel_size)
        self.up3 = Up(channels_depth_number[2], channels_depth_number[1] // factor, mode, kernel_size=kernel_size)
        self.up4 = Up(channels_depth_number[1], channels_depth_number[0], mode, kernel_size=kernel_size)
        self.outc = OutConv(channels_depth_number[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
