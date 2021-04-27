import torch
from torch import nn


class ConvBlock1(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, batch_norm_momentum=0.1):
		super().__init__()
		self.relu = nn.LeakyReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
		self.bn1 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x1 = self.relu(x)
		x2 = self.conv2(x1)
		x2 = self.bn2(x2)
		out = x1 + x2
		return out

class ConvBlock2(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size_1=3, kernel_size_2=1, batch_norm_momentum=0.1):
		super().__init__()
		self.relu = nn.LeakyReLU(inplace=True)
		if kernel_size_2 == 3:
			padding = 1
		else:
			padding = 0
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size_1, stride=2, padding=1)
		self.bn1 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size_1, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
		self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size_2, stride=2, padding=padding)
		self.bn3 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)


	def forward(self, x):
		tmp = x.clone()
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		tmp = self.conv3(tmp)
		tmp = self.bn3(tmp)
		out = x + tmp
		return out


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            #self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)

            #self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x):
    	x = self.up(x)
    	return x



class BreastCNN(nn.Module):
	def __init__(self, in_channels, out_channels, batch_norm_momentum=0.1, middle_channels=[16, 32, 64]):
		super().__init__()

		self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
		self.up2 = nn.Upsample(scale_factor=4, mode='nearest')
		#self.up1 = Up(middle_channels[1], 1, scale_factor=2)
		#self.up2 = Up(middle_channels[2], 1, scale_factor=4)


		self.blockDown1_1 = ConvBlock1(in_channels, middle_channels[0], batch_norm_momentum=batch_norm_momentum)
		self.blockDown1_2 = ConvBlock1(middle_channels[0], middle_channels[0], batch_norm_momentum=batch_norm_momentum)

		self.blockDown2_1 = ConvBlock2(middle_channels[0], middle_channels[1], batch_norm_momentum=batch_norm_momentum)
		self.blockDown2_2 = ConvBlock1(middle_channels[1], middle_channels[1], batch_norm_momentum=batch_norm_momentum)
		self.blockDown2_3 = ConvBlock1(middle_channels[1], middle_channels[1], batch_norm_momentum=batch_norm_momentum)

		self.blockDown3_1 = ConvBlock2(middle_channels[1], middle_channels[2], kernel_size_2=3, batch_norm_momentum=batch_norm_momentum)
		self.blockDown3_2 = ConvBlock1(middle_channels[2], middle_channels[2], batch_norm_momentum=batch_norm_momentum)
		self.blockDown3_3 = ConvBlock1(middle_channels[2], middle_channels[2], batch_norm_momentum=batch_norm_momentum)

		self.conv_1x1_1 = nn.Conv2d(middle_channels[0], out_channels, kernel_size=1)
		self.bn_1x1_1 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
		self.conv_1x1_2 = nn.Conv2d(middle_channels[1], out_channels, kernel_size=1)
		self.bn_1x1_2 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
		self.conv_1x1_3 = nn.Conv2d(middle_channels[2], out_channels, kernel_size=1)
		self.bn_1x1_3 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)

		self.out_conv = nn.Conv2d(3, out_channels, kernel_size=1)
		self.out_bn = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
		self.out_relu = nn.ReLU(inplace=True)



	def forward(self, x):
		x = self.blockDown1_1(x)
		x = self.blockDown1_2(x)

		x1 = x.clone()
		x1 = self.conv_1x1_1(x1)
		x1 = self.bn_1x1_1(x1)

		x = self.blockDown2_1(x)
		x = self.blockDown2_2(x)
		x = self.blockDown2_3(x)

		x2 = x.clone()
		x2 = self.conv_1x1_2(x2)
		x2 = self.bn_1x1_2(x2)
		x2 = self.up1(x2)

		x = self.blockDown3_1(x)
		x = self.blockDown3_2(x)
		x = self.blockDown3_3(x)
		x = self.conv_1x1_3(x)
		x = self.bn_1x1_3(x)
		x = self.up2(x)

		out = self.out_conv(torch.cat([x, x1, x2], 1))
		out = self.out_bn(out)
		out = self.out_relu(out)
		return out