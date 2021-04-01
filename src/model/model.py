import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------- #
#                                  Unet model                                       #
#    This piece of code was taken from https://github.com/milesial/Pytorch-UNet     #
# --------------------------------------------------------------------------------- #


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, use_relu=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_relu:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
        )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, use_relu=True, stride=1, padding=1, residual_block=False, 
        batch_norm_momentum=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_relu:
            self.double_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels, momentum=batch_norm_momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )
        else:
            self.double_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels, momentum=batch_norm_momentum),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )
        if residual_block:
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
            )
        else:
            self.skip = None

    def forward(self, x):
        if self.skip is None:
            return self.double_conv(x)
        else:
            return self.double_conv(x) + self.skip(x)

class DoubleConvIni(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, use_relu=True, padding=1, residual_block=False,
        batch_norm_momentum=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_relu:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )
        if residual_block:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        else:
            self.skip = None

    def forward(self, x):
        if self.skip is None:
            return self.double_conv(x)
        else:
            return self.double_conv(x) + self.skip(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, residual_block=False, batch_norm_momentum=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            DoubleConv2(in_channels, out_channels, kernel_size=kernel_size, stride=2, residual_block=residual_block, batch_norm_momentum=batch_norm_momentum)
        )

    def forward(self, x):
        #print("Down forward")
        #print(x.size())
        tmp =  self.maxpool_conv(x)
        #print(tmp.size())
        #print()
        return tmp


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mode='bilinear', kernel_size=3, residual_block=False, batch_norm_momentum=0.1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2(in_channels, out_channels, kernel_size=kernel_size, residual_block=residual_block, batch_norm_momentum=batch_norm_momentum)
        elif mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2) # nearest neighbor, not working
            self.conv = DoubleConv2(in_channels, out_channels, kernel_size=kernel_size, residual_block=residual_block, batch_norm_momentum=batch_norm_momentum)
        else:
            print("Not implemented")
            exit(0)


    def forward(self, x1, x2):
        #print("Up Forward")
        #print(x1.size())
        x1 = self.up(x1)
        #print(x1.size())
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        #print(x1.size())
        #print(x2.size())
        #print(x.size())
        #print()
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, mode='bilinear', channels_depth_number=(64, 128, 256, 512, 1024), 
        kernel_size=3, use_relu=True, residual_block=False, batch_norm_momentum=0.1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        # encoder
        self.inc = DoubleConvIni(n_channels, channels_depth_number[0], kernel_size=kernel_size, residual_block=residual_block)
        self.down1 = Down(channels_depth_number[0], channels_depth_number[1], kernel_size=kernel_size, residual_block=residual_block)
        self.down2 = Down(channels_depth_number[1], channels_depth_number[2], kernel_size=kernel_size, residual_block=residual_block)
        self.down3 = Down(channels_depth_number[2], channels_depth_number[3], kernel_size=kernel_size, residual_block=residual_block)
        if mode == 'bilinear' or mode == 'nearest':
            factor = 2
        else:
            factor = 1
        # bridge
        self.down4 = Down(channels_depth_number[3], channels_depth_number[4] // factor, kernel_size=kernel_size, residual_block=residual_block)
        # decoder
        self.up1 = Up(channels_depth_number[4], channels_depth_number[3] // factor, mode, kernel_size=kernel_size, residual_block=residual_block)
        self.up2 = Up(channels_depth_number[3], channels_depth_number[2] // factor, mode, kernel_size=kernel_size, residual_block=residual_block)
        self.up3 = Up(channels_depth_number[2], channels_depth_number[1] // factor, mode, kernel_size=kernel_size, residual_block=residual_block)
        self.up4 = Up(channels_depth_number[1], channels_depth_number[0], mode, kernel_size=kernel_size, residual_block=residual_block)
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
        #exit(0)
        return logits
