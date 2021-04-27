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
		if kernel_size_1 == kernel_size_2:
			self.same_kernels = True
		else:
			self.same_kernels = False

		self.relu = nn.LeakyReLU(inplace=True)

		if middle_channels is None:
			middle_channels	= out_channels

		if kernel_size_1 == 3:
			padding_1 = 1
		elif kernel_size_1 == 5:
			padding_1 = 2
		elif kernel_size_1 == 7:
			padding_1 = 3
		elif kernel_size_1 == 1:
			padding_1 = 0

		if kernel_size_2 == 3:
			padding_2 = 1
		elif kernel_size_2 == 5:
			padding_2 = 2
		elif kernel_size_2 == 7:
			padding_2 = 3
		elif kernel_size_2 == 1:
			padding_2 = 0

		# nxn convolutions 1
		if self.same_kernels:
			factor = 2
		else:
			factor = 1
		self.conv_nxn_11 = nn.Conv2d(in_channels, middle_channels*factor, kernel_size=kernel_size_1, padding=padding_1)
		self.conv_nxn_12 = nn.Conv2d(middle_channels*factor, out_channels*factor, kernel_size=kernel_size_1, padding=padding_1)
		self.skip_nxn_1 = nn.Conv2d(in_channels, out_channels*factor, kernel_size=kernel_size_1, padding=padding_1)

		if not self.same_kernels:
			# nxn convolutions 2
			self.conv_nxn_21 = nn.Conv2d(in_channels, middle_channels, kernel_size=kernel_size_2, padding=padding_2)
			self.conv_nxn_22 = nn.Conv2d(middle_channels, out_channels, kernel_size=kernel_size_2, padding=padding_2)
			self.skip_nxn_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size_2, padding=padding_2)

		self.bn_nxn_11 = nn.BatchNorm2d(middle_channels*factor, momentum=batch_norm_momentum)
		self.bn_nxn_12 = nn.BatchNorm2d(out_channels*factor, momentum=batch_norm_momentum)

		if not self.same_kernels:
			self.bn_nxn_21 = nn.BatchNorm2d(middle_channels, momentum=batch_norm_momentum)
			self.bn_nxn_22 = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)

	def forward_1(self, x):
		res_1 = x.clone()
		res_2 = x.clone()
		#conv nxn 1
		out_1 = self.conv_nxn_11(x)
		out_1 = self.bn_nxn_11(out_1)
		out_1 = self.relu(out_1)
		out_1 = self.conv_nxn_12(out_1)
		out_1 = self.bn_nxn_12(out_1)
		
		if res_1.size() == out_1.size():
			out_1 += res_1
		else:
			out_1 += self.skip_nxn_1(x)
		out_1 = self.relu(out_1) 

		#conv nxn 2
		out_2 = self.conv_nxn_21(x)
		out_2 = self.bn_nxn_21(out_2)
		out_2 = self.relu(out_2)
		out_2 = self.conv_nxn_22(out_2)
		out_2 = self.bn_nxn_22(out_2)
		
		# concatenate result
		if res_2.size() == out_2.size():
			out_2 += res_2
		else:
			out_2 += self.skip_nxn_2(x)
		out_2 = self.relu(out_2) 

		out = torch.cat([out_1, out_2], 1)
		return out

	def forward_2(self, x):
		out_1 = self.conv_nxn_11(x)
		out_1 = self.bn_nxn_11(out_1)
		out_1 = self.relu(out_1)
		out_1 = self.conv_nxn_12(out_1)
		out_1 = self.bn_nxn_12(out_1)
		out_1 += self.skip_nxn_1(x)
		out_1 = self.relu(out_1)
		return out_1 

	def forward(self, x):
		if self.same_kernels:
			return self.forward_2(x)
		else:
			return self.forward_1(x)


class InceptionUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, nb_filter=[32, 64, 128, 256, 512], batch_norm_momentum=0.1, kernel_size_1=[7,5,3,3,3], kernel_size_2=[5,3,3,3,1]):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.activation_relu = nn.ReLU(inplace=True)
        self.output_bn = nn.BatchNorm2d(num_classes, momentum=batch_norm_momentum)

        # Encoder
        self.conv0_0 = InceptionBlock(input_channels, nb_filter[0]//2, nb_filter[0]//2, 
        	kernel_size_1=kernel_size_1[0], kernel_size_2=kernel_size_2[0], batch_norm_momentum=0.1)
        self.conv1_0 = InceptionBlock(nb_filter[0], nb_filter[1]//2, nb_filter[1]//2, 
        	kernel_size_1=kernel_size_1[1], kernel_size_2=kernel_size_2[1], batch_norm_momentum=0.1)
        self.conv2_0 = InceptionBlock(nb_filter[1], nb_filter[2]//2, nb_filter[2]//2, 
        	kernel_size_1=kernel_size_1[2], kernel_size_2=kernel_size_2[2], batch_norm_momentum=0.1)
        self.conv3_0 = InceptionBlock(nb_filter[2], nb_filter[3]//2, nb_filter[3]//2, 
        	kernel_size_1=kernel_size_1[3], kernel_size_2=kernel_size_2[3], batch_norm_momentum=0.1)
        # bridge
        self.conv4_0 = InceptionBlock(nb_filter[3], nb_filter[4]//2, nb_filter[4]//2, 
        	kernel_size_1=kernel_size_1[4], kernel_size_2=kernel_size_2[4], batch_norm_momentum=0.1)
        # decoder
        self.conv3_1 = InceptionBlock(nb_filter[3]+nb_filter[4], nb_filter[3]//2, nb_filter[3]//2, 
        	kernel_size_1=kernel_size_1[3], kernel_size_2=kernel_size_2[3], batch_norm_momentum=0.1)
        self.conv2_2 = InceptionBlock(nb_filter[2]+nb_filter[3], nb_filter[2]//2, nb_filter[2]//2, 
        	kernel_size_1=kernel_size_1[2], kernel_size_2=kernel_size_2[2], batch_norm_momentum=0.1)
        self.conv1_3 = InceptionBlock(nb_filter[1]+nb_filter[2], nb_filter[1]//2, nb_filter[1]//2, 
        	kernel_size_1=kernel_size_1[1], kernel_size_2=kernel_size_2[1], batch_norm_momentum=0.1)
        self.conv0_4 = InceptionBlock(nb_filter[0]+nb_filter[1], nb_filter[0]//2, nb_filter[0]//2, 
        	kernel_size_1=kernel_size_1[0], kernel_size_2=kernel_size_2[0], batch_norm_momentum=0.1)

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
        output = self.output_bn(output)
        output = self.activation_relu(output)
        #exit(0)
        return output












