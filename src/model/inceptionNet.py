import torch
from torch import nn

class InceptionBlock(nn.Module):
	def __init__(self, in_channels, out_channels, batch_norm_momentum=0.1, use_maxpool=True, kernel_asym_dim=None, up=False):
		super().__init__()
		self.upsample = up
		self.reduce = True
		self.use_maxpool = use_maxpool
		if in_channels == out_channels:
			self.reduce = False
		reduce_channels = out_channels
		if kernel_asym_dim is not None:
			pad = (kernel_asym_dim - 1) // 2
			kernel_size1x1 = (1, kernel_asym_dim)
			padding1x1 = (0, pad)
			kernel_size3x3 = (3, kernel_asym_dim)
			padding3x3 = (1, pad)
			kernel_size5x5 = (5, kernel_asym_dim)
			padding5x5 = (2, pad)
			max_pool_size = (3, kernel_asym_dim)
			padding_maxpool = (1, pad)
		else:
			if self.upsample == True:
				self.upsample = False
			kernel_size1x1 = 1
			padding1x1 = 0
			kernel_size3x3 = 3
			padding3x3 = 1
			kernel_size5x5 = 5
			padding5x5 = 2
			max_pool_size = 3
			padding_maxpool = 1
		if self.reduce:
			self.conv_reduce3x3 = nn.Conv2d(in_channels, reduce_channels, kernel_size=kernel_size1x1, padding=padding1x1)
			self.conv_reduce5x5 = nn.Conv2d(in_channels, reduce_channels, kernel_size=kernel_size1x1, padding=padding1x1)
			if self.use_maxpool:
				self.maxpool_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size1x1, padding=padding1x1)
		self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size1x1, padding=padding1x1)
		self.conv3x3 = nn.Conv2d(reduce_channels, out_channels, kernel_size=kernel_size3x3, padding=padding3x3)
		self.conv5x5 = nn.Conv2d(reduce_channels, out_channels, kernel_size=kernel_size5x5, padding=padding5x5)
		if self.use_maxpool:
			self.maxpool = nn.MaxPool2d(max_pool_size, padding=padding_maxpool, stride=1)
		if self.reduce:
			self.bn = nn.BatchNorm2d(in_channels, momentum=batch_norm_momentum)
		else:
			if self.use_maxpool:
				self.bn = nn.BatchNorm2d(4*out_channels, momentum=batch_norm_momentum)
			else:
				self.bn = nn.BatchNorm2d(3*out_channels, momentum=batch_norm_momentum)
		self.relu = nn.LeakyReLU(inplace=True)
		print(self.upsample)
		if self.upsample:
			self.up = nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True)

	def forward(self, x):
		if self.upsample:
			x = self.up(x)
		x1 = self.conv1x1(x)

		if self.reduce:
			x2 = self.conv_reduce3x3(x)
			x2 = self.conv3x3(x2)
			x3 = self.conv_reduce5x5(x)
			x3 = self.conv5x5(x3)
			if self.use_maxpool:
				x4 = self.maxpool(x)
				x4 = self.maxpool_reduce(x4)
		else:
			x2 = self.conv3x3(x)
			x3 = self.conv5x5(x)
			if self.use_maxpool:
				x4 = self.maxpool(x)
		if self.use_maxpool:
			out = torch.cat([x1, x2, x3, x4], 1)
		else:
			out = torch.cat([x1, x2, x3], 1)
		out = self.bn(out)
		out = self.relu(out)
		return out


class InceptionNet(nn.Module):
	def __init__(self, in_channels, out_channels, inception_channels, n_inception_blocks=3, batch_norm_momentum=0.1,
		use_maxpool=True, kernel_asym_dim=None, up=[False, False, False]):
		super().__init__()
		if len(up) != n_inception_blocks:
			raise RuntimeError("Matching in upsample and inception blocks does not work!")
		factor = 4
		if not use_maxpool:
			factor = 3
		# self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
		if kernel_asym_dim is not None:
			self.conv2 = nn.Conv2d(in_channels, inception_channels, kernel_size=(3, kernel_asym_dim), padding=(1, (kernel_asym_dim - 1) // 2))
		else:
			self.conv2 = nn.Conv2d(in_channels, inception_channels, kernel_size=3, padding=1)

		self.inceptionBlock1 = InceptionBlock(inception_channels, inception_channels, 
			batch_norm_momentum=batch_norm_momentum, use_maxpool=use_maxpool, kernel_asym_dim=kernel_asym_dim, up=up[0])
		self.inceptionBlocks = []
		for i in range(n_inception_blocks - 1):
			inceptionBlock = InceptionBlock(factor*inception_channels, inception_channels, 
				batch_norm_momentum=batch_norm_momentum, use_maxpool=use_maxpool, kernel_asym_dim=kernel_asym_dim, up=up[i + 1])
			self.inceptionBlocks.append(inceptionBlock)
			self.add_module('InceptionBlock-%d' % i, inceptionBlock)
		self.bn = nn.BatchNorm2d(inception_channels)
		self.relu = nn.LeakyReLU(inplace=True)
		self.out_conv = nn.Conv2d(factor*inception_channels, out_channels, kernel_size=1)
		self.out_bn = nn.BatchNorm2d(out_channels)
		self.out_relu = nn.ReLU(inplace=True)
		print("HEREREERERER")

	def forward(self, x):
		# x = self.conv1(x)
		# x = self.relu(x)
		x = self.conv2(x)
		x = self.bn(x)
		x = self.relu(x)
		x = self.inceptionBlock1(x)
		for inceptionBlock in self.inceptionBlocks:
			x = inceptionBlock(x)
		x = self.out_conv(x)
		x = self.out_bn(x)
		x = self.out_relu(x)
		return x

