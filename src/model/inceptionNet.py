import torch
from torch import nn

class InceptionBlock(nn.Module):
	def __init__(self, in_channels, out_channels, batch_norm_momentum=0.1, use_maxpool=True):
		super().__init__()
		self.reduce = True
		self.use_maxpool = use_maxpool
		if in_channels == out_channels:
			self.reduce = False
		reduce_channels = out_channels
		if self.reduce:
			self.conv_reduce3x3 = nn.Conv2d(in_channels, reduce_channels, kernel_size=1)
			self.conv_reduce5x5 = nn.Conv2d(in_channels, reduce_channels, kernel_size=1)
			if self.use_maxpool:
				self.maxpool_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
		self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
		self.conv3x3 = nn.Conv2d(reduce_channels, out_channels, kernel_size=3, padding=1)
		self.conv5x5 = nn.Conv2d(reduce_channels, out_channels, kernel_size=5, padding=2)
		if self.use_maxpool:
			self.maxpool = nn.MaxPool2d(3, padding=1, stride=1)
		if self.reduce:
			self.bn = nn.BatchNorm2d(in_channels, momentum=batch_norm_momentum)
		else:
			if self.use_maxpool:
				self.bn = nn.BatchNorm2d(4*out_channels, momentum=batch_norm_momentum)
			else:
				self.bn = nn.BatchNorm2d(3*out_channels, momentum=batch_norm_momentum)
		self.relu = nn.LeakyReLU(inplace=True)

	def forward(self, x):
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
		use_maxpool=True):
		super().__init__()
		factor = 4
		if not use_maxpool:
			factor = 3
		self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(in_channels, inception_channels, kernel_size=3, padding=1)
		self.inceptionBlock1 = InceptionBlock(inception_channels, inception_channels, 
			batch_norm_momentum=batch_norm_momentum, use_maxpool=use_maxpool)
		self.inceptionBlocks = []
		for i in range(n_inception_blocks - 1):
			inceptionBlock = InceptionBlock(factor*inception_channels, inception_channels, 
				batch_norm_momentum=batch_norm_momentum, use_maxpool=use_maxpool)
			self.inceptionBlocks.append(inceptionBlock)
			self.add_module('InceptionBlock-%d' % i, inceptionBlock)
		self.bn = nn.BatchNorm2d(inception_channels)
		self.relu = nn.LeakyReLU(inplace=True)
		self.out_conv = nn.Conv2d(factor*inception_channels, out_channels, kernel_size=1)
		self.out_bn = nn.BatchNorm2d(out_channels)
		self.out_relu = nn.ReLU(inplace=True)

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

