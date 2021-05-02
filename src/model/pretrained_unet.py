import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------- #
#                             Pretrained Unet model                                 #
# The pretrained model is downloaded from https://github.com/milesial/Pytorch-UNet  #
# --------------------------------------------------------------------------------- #


class PretrainedUNet(nn.Module):

    def __init__(self, n_channels, unfreezed_layers: list):
        super().__init__()

        # Download pre-trained model from github
        if torch.cuda.is_available():
            model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True)
        else:
            model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=False)
            state_dict = torch.hub.load_state_dict_from_url(
                'https://github.com/milesial/Pytorch-UNet/releases/download/v1.0/unet_carvana_scale1_epoch5.pth',
                map_location="cpu"
            )
            model.load_state_dict(state_dict)

        self.model = model
        self.n_channels = n_channels
        #self.out_bn = nn.BatchNorm2d(1, momen)
        self.relu = nn.ReLU(inplace=True)
        # Intial convolution to transform input image from 1 to 3 channels
        self.initial_conv = nn.Conv2d(self.n_channels, 3, kernel_size=1)

        # Freeze unwanted gradients
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in unfreezed_layers):
                pass
            else:
                param.requires_grad = False

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.model.forward(x)
        x = self.relu(x)
        return x


if __name__ == "__main__":
    if torch.cuda.is_available():
        model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True)
    else:
        model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            'https://github.com/milesial/Pytorch-UNet/releases/download/v1.0/unet_carvana_scale1_epoch5.pth',
            map_location="cpu"
        )
        model.load_state_dict(state_dict)

    print(model)

    for name, param in model.named_parameters():
        print(name)
