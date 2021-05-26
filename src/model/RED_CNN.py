import torch
import torch.nn as nn
import os

# --------------------------------------------------------------------------------- #
#                                Pretrained Red-CNN                                 #
#                        https://github.com/SSinyu/RED-CNN                          #
# --------------------------------------------------------------------------------- #


class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5)
        self.up = nn.Upsample(scale_factor=(4, 1), mode='bilinear', align_corners=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        x = self.up(x)
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out


class PretrainedREDCNN(nn.Module):

    def __init__(self, unfreezed_layers: list):
        super().__init__()

        model = RED_CNN()

        # Download pre-trained model
        iter_: int = 100000
        model_folder = "model"
        save_path: str = "save"
        f = os.path.join(model_folder, save_path, 'REDCNN_{}iter.ckpt'.format(iter_))

        if torch.cuda.is_available():
            model.load_state_dict(state_dict=torch.load(f))
        else:
            model.load_state_dict(state_dict=torch.load(f, map_location="cpu"))

        self.model = model

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
    model = RED_CNN()

    iter_: int = 100000
    save_path: str = "save/"
    f = os.path.join(save_path, 'REDCNN_{}iter.ckpt'.format(iter_))

    if torch.cuda.is_available():
        model.load_state_dict(state_dict=torch.load(f))
    else:
        model.load_state_dict(state_dict=torch.load(f, map_location="cpu"))

    print(model)

    for name, param in model.named_parameters():
        print(name)
