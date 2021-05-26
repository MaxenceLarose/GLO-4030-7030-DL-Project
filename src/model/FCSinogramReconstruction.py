import torch
from torch import nn
from .nestedUnet import VGGBlock
from .unet import UNet


class AsymetricConv1(nn.Module):
	def __init__(self, in_channels, out_channels, up=False):
		super().__init__()
		self.upsample = up
		self.asymconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 257), padding=(1, 0), stride=(1,1))
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.asymconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 257), padding=(1, 0), stride=(1,1))
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.up = nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True)

	def forward(self, x):
		x = self.asymconv1(x)
		#print(x.size())
		# exit(0)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.asymconv2(x)
		#print(x.size())
		x = self.bn2(x)
		x = self.relu(x)
		if self.upsample:
			out = self.up(x)
		else:
			out = x
		return out

class AsymetricConvUp(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=(3,7), padding=(1,3), stride=(1,1), up='ConvTranspose2d'):
		super().__init__()
		self.up = up
		if self.up == "bilinear":
			self.asymconv1 = nn.Sequential(
				nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=True),
				nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=(1,1))
				)
		elif self.up == "ConvTranspose2d":
			self.asymconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(kernel_size[0], kernel_size[1]-1), stride=(1,2))
		self.bn1 = nn.BatchNorm2d(out_channels)

		if self.up == "ConvTranspose2d":
			self.asymconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=(0, 1), stride=(1,1))
		else:
			self.asymconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=(1,1))
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU()
		
	def forward(self, x):
		x = self.asymconv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.asymconv2(x)
		x = self.bn2(x)
		out = self.relu(x)
		return out

class AsymetricConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=(3,7), padding=(1,3), stride=(1,1)):
		super().__init__()
		self.asymconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.asymconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU()
		
	def forward(self, x):
		x = self.asymconv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.asymconv2(x)
		x = self.bn2(x)
		out = self.relu(x)
		return out

class AsymetricConvDown(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=(3,7), padding=(1,3), stride=(1,2)):
		super().__init__()
		self.asymconv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.asymconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=(1,1))
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU()
		
	def forward(self, x):
		x = self.asymconv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.asymconv2(x)
		x = self.bn2(x)
		out = self.relu(x)
		return out



class SinogramInterpolator(nn.Module):
	def __init__(self, in_channels, mid_channels, out_channels):
		super().__init__()
		self.out_bn = nn.BatchNorm2d(out_channels)
		self.out_relu = nn.ReLU()
		self.up = nn.Upsample(scale_factor=(4, 1), mode='bicubic', align_corners=False)
		#self.upconv = nn.ConvTranspose2d(1, 1, kernel_size=(4, 11), padding=(0,5), stride=(4, 1))
		#mid_channels = 1
		self.in_conv = AsymetricConv(1, mid_channels, kernel_size=(3, 3), padding=(1,1))
		self.asymconv1 = AsymetricConv(mid_channels, mid_channels, kernel_size=(3, 11), padding=(1,5))
		self.asymconv2 = AsymetricConvDown(mid_channels, mid_channels, kernel_size=(3, 9), padding=(1,4))
		self.asymconv3 = AsymetricConvDown(mid_channels, mid_channels, kernel_size=(3,7), padding=(1,3))
		self.asymconv4 = AsymetricConvDown(mid_channels, mid_channels, kernel_size=(3,5), padding=(1,2))
		self.asymconv5 = AsymetricConv(mid_channels, mid_channels, kernel_size=(3,3), padding=(1,1))
		self.asymconv6 = AsymetricConvUp(2*mid_channels, mid_channels, kernel_size=(3,5), padding=(1,2))
		self.asymconv7 = AsymetricConvUp(2*mid_channels, mid_channels, kernel_size=(3,7), padding=(1,3))
		self.asymconv8 = AsymetricConvUp(2*mid_channels, mid_channels, kernel_size=(3,9), padding=(1,4))
		self.asymconv9 = AsymetricConv(2*mid_channels, mid_channels, kernel_size=(3,11), padding=(1,5))
		self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
		#self.x_rep = torch.zeros(1,1, self.n_proj_out, self.n_pix_y).cuda()

	def forward(self, x):
		# self.x_rep.grad = None
		# self.x_rep.fill_(0)
		#print(self.n_pix_y)
		# x_upconv = self.upconv(x)
		#x = self.up(x)
		# exit(0)
		res1 = x
		#x = torch.cat([x, x_upconv], 1)
		# x = self.in_conv(x)
		# x = self.asymconv1(x)
		# res2 = x
		# x = self.asymconv2(x)
		# res3 = x
		# x = self.asymconv3(x)
		# res4 = x
		# x = self.asymconv4(x)
		# res5 = x
		# x = self.asymconv5(x)
		# x = torch.cat([x, res5], 1)
		# x = self.asymconv6(x)
		# x = torch.cat([x, res4], 1)
		# x = self.asymconv7(x)
		# x = torch.cat([x, res3], 1)
		# x = self.asymconv8(x)
		# x = torch.cat([x, res2], 1)
		# x = self.asymconv9(x)
		# x = self.out_conv(x)
		# x = self.out_bn(x)
		# x += res1
		#exit(0)
		# out = self.out_relu(x)
		return res1
