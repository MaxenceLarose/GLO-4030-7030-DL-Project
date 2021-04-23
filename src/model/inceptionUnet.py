import torch
from torch import nn

class VGGBlock(nn.Module):
	def __init__(self, in_channels, middle_channels, out_channels, kernel_size=3, batch_norm_momentum=0.1):
		super().__init__()
		self.relu = nn.LeakyReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=kernel_size, padding=1)
		self.bn1 = nn.BatchNorm2d(middle_channels, momentum=batch_norm_momentum)
		self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=kernel_size, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		return out

class InceptionBlock(nn.Module):
	def __init__(self, in_channels, out_channels, middle_channels=None, reduce_channels=True, kernel_size_1=3, kernel_size_2=5, batch_norm_momentum=0.1):
		super().__init__()
		self.relu = nn.LeakyReLU(inplace=True)
		self.reduce_channels = reduce_channels

		if self.reduce_channels:
			self.conv1x1_reduce = nn.Conv2d(2*out_channels, out_channels, kernel_size=1)
		if middle_channels is None:
			middle_channels	= out_channels
		# 3x3 convolutions
		self.conv3x3_1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
		self.conv3x3_2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
		self.skip_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

		# 5x5 convolutions
		self.conv5x5_1 = nn.Conv2d(in_channels, middle_channels, kernel_size=5, padding=2)
		self.conv5x5_2 = nn.Conv2d(middle_channels, out_channels, kernel_size=5, padding=2)
		self.skip_5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
		# 7x7 convolutions
		# self.conv7x7_1 = nn.Conv2d(in_channels, middle_channels, kernel_size=7, padding=3)
		# self.conv7x7_2 = nn.Conv2d(middle_channels, out_channels, kernel_size=7, padding=3)

		self.bn_3x3_1 = nn.BatchNorm2d(middle_channels, momentum=batch_norm_momentum)
		self.bn_3x3_2 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
		self.bn_5x5_1 = nn.BatchNorm2d(middle_channels, momentum=batch_norm_momentum)
		self.bn_5x5_2 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
		# self.bn_7x7_1 = nn.BatchNorm2d(middle_channels, momentum=batch_norm_momentum)
		# self.bn_7x7_2 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
		self.bn_1x1 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)

	def forward(self, x):

		#if reduce_channels is not None:
		#	x = self.conv1x1_reduce(x)
		#print(x)
		#tmp = x.clone()
		#res_3x3 = x[:, :x.size()[1]//2, :, :].clone()
		#res_5x5 = x[:, x.size()[1]//2:, :, :].clone()
		res_3x3 = x.clone()
		res_5x5 = x.clone()
		out3x3 = self.conv3x3_1(x)
		out3x3 = self.bn_3x3_1(out3x3)
		out3x3 = self.relu(out3x3)
		out3x3 = self.conv3x3_2(out3x3)
		out3x3 = self.bn_3x3_2(out3x3)
		# out3x3 = self.relu(out3x3 + skip_3x3)
		#print(res_3x3.size())
		#print(out3x3.size())
		#print()
		if res_3x3.size() == out3x3.size():
			out3x3 += res_3x3
		else:
			out3x3 += self.skip_3x3(x)
		out3x3 = self.relu(out3x3) 

		#print(x)
		#exit(0)
		out5x5 = self.conv5x5_1(x)
		out5x5 = self.bn_5x5_1(out5x5)
		out5x5 = self.relu(out5x5)
		out5x5 = self.conv5x5_2(out5x5)
		out5x5 = self.bn_5x5_2(out5x5)
		# out5x5 = self.relu(out5x5 + skip_5x5)
		#out5x5 = self.relu(out5x5)
		if res_5x5.size() == out5x5.size():
			out5x5 += res_5x5
		else:
			out5x5 += self.skip_5x5(x)
		out5x5 = self.relu(out5x5) 


		out = torch.cat([out3x3, out5x5], 1)
		#print(out.size())
		# if self.reduce_channels:
		# 	out = self.conv1x1_reduce(out)
			# out = self.bn_1x1(out)
			# out = self.relu(out)

		return out


class InceptionUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, nb_filter=[32, 64, 128, 256, 512], batch_norm_momentum=0.1):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv0_0 = InceptionBlock(input_channels, nb_filter[0]//2, nb_filter[0]//2, batch_norm_momentum=0.1)
        self.conv1_0 = InceptionBlock(nb_filter[0], nb_filter[1]//2, nb_filter[1]//2, batch_norm_momentum=0.1)
        self.conv2_0 = InceptionBlock(nb_filter[1], nb_filter[2]//2, nb_filter[2]//2, batch_norm_momentum=0.1)
        self.conv3_0 = InceptionBlock(nb_filter[2], nb_filter[3]//2, nb_filter[3]//2, batch_norm_momentum=0.1)
        self.conv4_0 = InceptionBlock(nb_filter[3], nb_filter[4]//2, nb_filter[4]//2, batch_norm_momentum=0.1)

        self.conv3_1 = InceptionBlock(nb_filter[3]+nb_filter[4], nb_filter[3]//2, nb_filter[3]//2, batch_norm_momentum=0.1)
        self.conv2_2 = InceptionBlock(nb_filter[2]+nb_filter[3], nb_filter[2]//2, nb_filter[2]//2, batch_norm_momentum=0.1)
        self.conv1_3 = InceptionBlock(nb_filter[1]+nb_filter[2], nb_filter[1]//2, nb_filter[1]//2, batch_norm_momentum=0.1)
        self.conv0_4 = InceptionBlock(nb_filter[0]+nb_filter[1], nb_filter[0]//2, nb_filter[0]//2, batch_norm_momentum=0.1)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        #exit(0)
        return output












