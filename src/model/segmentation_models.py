import logging
import numpy as np

import torch.nn as nn

import segmentation_models_pytorch as smp

# --------------------------------------------------------------------------------- #
#               Pre-trained models from segmentation_models_pytorch                 #
#  Useful resources :                                                               #
#  https://github.com/qubvel/segmentation_models.pytorch                            #
# --------------------------------------------------------------------------------- #


class UNetSMP(nn.Module):

    def __init__(self,
                 unfreezed_layers: list,
                 in_channels: int,
                 encoder_depth: int = 5,
                 encoder: str = "resnet34",
                 encoder_weights: str = "imagenet",
                 activation: str = "sigmoid",
                 ):
        super().__init__()

        self.model = smp.Unet(encoder_name=encoder,
                              encoder_depth=encoder_depth,
                              encoder_weights=encoder_weights,
                              in_channels=in_channels,
                              activation=activation,
                              )

        # Freeze unwanted gradients
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in unfreezed_layers):
                pass
            else:
                param.requires_grad = False

    def forward(self, x):
        x = self.model.forward(x)

        return x


class UNetPlusPLus(nn.Module):

    def __init__(self,
                 unfreezed_layers: list,
                 in_channels: int,
                 encoder_depth: int = 5,
                 encoder: str = "resnet34",
                 decoder_channels: tuple = (256, 128, 64, 32, 16),
                 encoder_weights: str = "imagenet",
                 activation: str = "sigmoid"
                 ):
        super().__init__()

        self.model = smp.UnetPlusPlus(encoder_name=encoder,
                                      encoder_depth=encoder_depth,
                                      encoder_weights=encoder_weights,
                                      in_channels=in_channels,
                                      decoder_channels=decoder_channels,
                                      activation=activation,
                                      )

        # Freeze unwanted gradients
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in unfreezed_layers):
                pass
            else:
                param.requires_grad = False

    def forward(self, x):
        x = self.model.forward(x)

        return x


if __name__ == "__main__":
    encoder = "resnet34"
    encoder_weights = "imagenet"
    activation = 'sigmoid'  # Could be "logits" or "softmax2d" for multicalss segmentation

    model = smp.Unet(encoder_name=encoder,
                     encoder_depth=5,
                     encoder_weights=encoder_weights,
                     in_channels=1,
                     activation=activation,
                     )

    print(model)
