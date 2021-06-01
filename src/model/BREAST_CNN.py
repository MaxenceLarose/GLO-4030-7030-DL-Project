import torch
from torch import nn


class ConvBlock1(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm_momentum=0.1, norm="BN", num_groups=4):
		super().__init__()
		self.relu = nn.LeakyReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
		if norm == "BN":
			self.bn1 = nn.BatchNorm2d(out_channels, momentum=norm_momentum)
		elif norm == "GN" and num_groups != 0:
			self.bn1 = nn.GroupNorm(num_groups, out_channels)
		else:
			raise RuntimeError("wrong norm option: {}".format(norm))
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
		if norm == "BN":
			self.bn2 = nn.BatchNorm2d(out_channels, momentum=norm_momentum)
		elif norm == "GN" and num_groups != 0:
			self.bn2 = nn.GroupNorm(num_groups, out_channels)
		else:
			raise RuntimeError("wrong norm option: {}".format(norm))

	def forward(self, x):
		x = self.conv1(x)
		x1 = self.bn1(x)
		x1 = self.relu(x1)
		x1 = self.conv2(x1)
		x1 += x
		x1 = self.bn2(x1)
		out = self.relu(x1)
		#out = x1 + x2
		return out

class ConvBlock2(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, norm_momentum=0.1, norm="BN", num_groups=4):
		super().__init__()
		self.relu = nn.LeakyReLU(inplace=True)
		if kernel_size == 3:
			padding = 1
		elif kernel_size == 5:
			padding = 2
		else:
			padding = 0
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding)
		if norm == "BN":
			self.bn1 = nn.BatchNorm2d(out_channels, momentum=norm_momentum)
		elif norm == "GN" and num_groups != 0:
			self.bn1 = nn.GroupNorm(num_groups, out_channels)
		else:
			raise RuntimeError("wrong norm option: {}".format(norm))
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
		if norm == "BN":
			self.bn2 = nn.BatchNorm2d(out_channels, momentum=norm_momentum)
		elif norm == "GN" and num_groups != 0:
			self.bn2 = nn.GroupNorm(num_groups, out_channels)
		else:
			raise RuntimeError("wrong norm option: {}".format(norm))
		self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding)
		if norm == "BN":
			self.bn3 = nn.BatchNorm2d(out_channels, momentum=norm_momentum)
		elif norm == "GN" and num_groups != 0:
			self.bn3 = nn.GroupNorm(num_groups, out_channels)
		else:
			raise RuntimeError("wrong norm option: {}".format(norm))

	def forward(self, x):
		tmp = x
		tmp = self.conv3(tmp)
		tmp = self.bn3(tmp)
		#tmp = self.relu(tmp)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x += tmp
		x = self.bn2(x)
		out = self.relu(x)
		#out = x + tmp
		return out


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')
            #self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)

            #self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x):
    	x = self.up(x)
    	return x



class BreastCNN(nn.Module):
	def __init__(self, in_channels, out_channels, norm_momentum=0.1, middle_channels=[16, 32, 64], unet_arch=False, norm="BN", num_groups=4,
		sparse_sinogram_net=False, upsample_mode="bilinear"):
		super().__init__()
		self.sparse_sinogram_net = sparse_sinogram_net
		self.unet_arch = unet_arch
		if upsample_mode == "linear" or upsample_mode == "bilinear" or upsample_mode == "trilinear":
			align_corners = True
		else:
			align_corners = False
		if self.sparse_sinogram_net:
			self.up = nn.Upsample(scale_factor=(4,1), mode=upsample_mode, align_corners=align_corners)
		if upsample_mode == "linear" or upsample_mode == "bilinear" or upsample_mode == "trilinear" or upsample_mode == "bicubic"\
		or upsample_mode == "nearest":
			self.up1 = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=align_corners)
			if self.unet_arch:
				self.up2 = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=align_corners)
				self.up3 = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=align_corners)
			else:
				self.up2 = nn.Upsample(scale_factor=4, mode=upsample_mode, align_corners=align_corners)
		elif upsample_mode == "ConvTranspose2d":
			self.up1 = nn.ConvTranspose2d(middle_channels[1] , middle_channels[1], kernel_size=2, stride=2)
			if self.unet_arch:
				self.up2 = nn.ConvTranspose2d(middle_channels[2] , middle_channels[2], kernel_size=2, stride=2)
				self.up3 = nn.ConvTranspose2d(middle_channels[1] , middle_channels[1], kernel_size=2, stride=2)
			else:
				raise RuntimeError("Conv transpose 2d not implemented!")
				#self.up2 = nn.Upsample(scale_factor=4, mode=upsample_mode, align_corners=align_corners)
		else:
			raise RuntimeError("Upsample method not available!")
		print(upsample_mode)

	
		self.blockDown1_1 = ConvBlock1(in_channels, middle_channels[0], norm_momentum=norm_momentum, norm=norm, num_groups=num_groups)
		self.blockDown1_2 = ConvBlock1(middle_channels[0], middle_channels[0], norm_momentum=norm_momentum, norm=norm, num_groups=num_groups)
		#self.blockDown1_3 = ConvBlock1(middle_channels[0], middle_channels[0], norm_momentum=norm_momentum)

		self.blockDown2_1 = ConvBlock2(middle_channels[0], middle_channels[1], norm_momentum=norm_momentum, norm=norm, num_groups=num_groups)
		self.blockDown2_2 = ConvBlock1(middle_channels[1], middle_channels[1], norm_momentum=norm_momentum, norm=norm, num_groups=num_groups)
		self.blockDown2_3 = ConvBlock1(middle_channels[1], middle_channels[1], norm_momentum=norm_momentum, norm=norm, num_groups=num_groups)
		if self.unet_arch:
			self.blockDown2_4 = ConvBlock1(middle_channels[1] + middle_channels[0], middle_channels[0], norm_momentum=norm_momentum, norm=norm, num_groups=num_groups)


		self.blockDown3_1 = ConvBlock2(middle_channels[1], middle_channels[2], norm_momentum=norm_momentum, norm=norm, num_groups=num_groups)
		self.blockDown3_2 = ConvBlock1(middle_channels[2], middle_channels[2], norm_momentum=norm_momentum, norm=norm, num_groups=num_groups)
		self.blockDown3_3 = ConvBlock1(middle_channels[2], middle_channels[2], norm_momentum=norm_momentum, norm=norm, num_groups=num_groups)
		if self.unet_arch:
			self.blockDown3_4 = ConvBlock1(middle_channels[2] +  middle_channels[1], middle_channels[1], norm_momentum=norm_momentum, norm=norm, num_groups=num_groups)
			self.blockDown3_5 = ConvBlock1(middle_channels[1] +  middle_channels[0], middle_channels[0], norm_momentum=norm_momentum, norm=norm, num_groups=num_groups)


		self.conv_1x1_1 = nn.Conv2d(middle_channels[0], out_channels, kernel_size=1)
		self.bn_1x1_1 = nn.BatchNorm2d(out_channels, momentum=norm_momentum)
		if self.unet_arch:
			self.conv_1x1_2 = nn.Conv2d(middle_channels[0], out_channels, kernel_size=1)
		else:
			self.conv_1x1_2 = nn.Conv2d(middle_channels[1], out_channels, kernel_size=1)
		self.bn_1x1_2 = nn.BatchNorm2d(out_channels, momentum=norm_momentum)
		if self.unet_arch:
			self.conv_1x1_3 = nn.Conv2d(middle_channels[0], out_channels, kernel_size=1)
		else:
			self.conv_1x1_3 = nn.Conv2d(middle_channels[2], out_channels, kernel_size=1)
		self.bn_1x1_3 = nn.BatchNorm2d(out_channels, momentum=norm_momentum)

		# output
		self.out_conv = nn.Conv2d(3, out_channels, kernel_size=1)
		self.out_bn = nn.BatchNorm2d(out_channels, momentum=norm_momentum)
		self.out_relu = nn.ReLU(inplace=True)



	def forward(self, x):
		if self.sparse_sinogram_net:
			x = self.up(x)
			res = x
		# res = x
		x = self.blockDown1_1(x)
		x = self.blockDown1_2(x)
		#x = self.blockDown1_3(x)

		x1 = x
		if self.unet_arch:
			x1_cat = x
		x1 = self.conv_1x1_1(x1)
		x1 = self.bn_1x1_1(x1)

		x = self.blockDown2_1(x)
		x = self.blockDown2_2(x)
		x = self.blockDown2_3(x)

		x2 = x
		if self.unet_arch:
			x2_cat = x
			x2 = self.up1(x2)
			x2 = torch.cat([x2, x1_cat], 1)
			x2 = self.blockDown2_4(x2)
			x2 = self.conv_1x1_2(x2)
			x2 = self.bn_1x1_2(x2)
		else:
			x2 = self.conv_1x1_2(x2)
			x2 = self.bn_1x1_2(x2)
			x2 = self.up1(x2)

		x = self.blockDown3_1(x)
		x = self.blockDown3_2(x)
		x = self.blockDown3_3(x)

		if self.unet_arch:
			x = self.up2(x)
			x = torch.cat([x, x2_cat], 1)
			x = self.blockDown3_4(x)
			x = self.up3(x)
			x = torch.cat([x, x1_cat], 1)
			x = self.blockDown3_5(x)
			x = self.conv_1x1_3(x)
			x = self.bn_1x1_3(x)
		else:
			x = self.conv_1x1_3(x)
			x = self.bn_1x1_3(x)
			x = self.up2(x)

		out = self.out_conv(torch.cat([x, x1, x2], 1))
		if self.sparse_sinogram_net:
			out += res
		# out += res
		# out = self.out_bn(out)
		out = self.out_relu(out)
		return out
		#return [out, x, x2, x1]