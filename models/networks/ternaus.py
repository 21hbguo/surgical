
from torch import nn
import torch
from torch.nn import functional as F
import torchvision
import os

PRETRAIN_URLS = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}

PRETRAIN_FILES = {
    'vgg11': 'vgg11-bbd30ac9.pth',
    'vgg16': 'vgg16-397923af.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
}

def load_pretrained_weights(model, model_name='vgg16', pretrain_root='../pre_train_ckp/', strict=True):
    weight_file = PRETRAIN_FILES.get(model_name)
    if not weight_file:
        return False
    weight_path = os.path.join(pretrain_root, weight_file)
    state_dict = torch.load(weight_path, map_location='cpu')

    # Support torchvision full VGG checkpoints when loading into `.features`.
    if isinstance(model, nn.Sequential):
        model_keys = set(model.state_dict().keys())
        if model_keys and not any(key in model_keys for key in state_dict.keys()):
            feature_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('features.'):
                    new_key = key[len('features.'):]
                    feature_state_dict[new_key] = value
            if feature_state_dict:
                state_dict = feature_state_dict

    model.load_state_dict(state_dict, strict=strict)
    return True

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super().__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class TernausNet16(nn.Module):
    def __init__(
        self,
        in_chns=3,
        class_num=2,
        num_filters=32,
        pretrained=True,
        pretrain_root='../pre_train_ckp/',
    ):
        super().__init__()
        self.params = {
            'in_chns': in_chns,
            'class_num': class_num,
            'num_filters': num_filters,
            'pretrained': pretrained,
            'pretrain_root': pretrain_root,
        }
        self.class_num = class_num
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(weights=None).features
        if pretrained:
            load_pretrained_weights(self.encoder, 'vgg16', pretrain_root, strict=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0], self.relu, self.encoder[2], self.relu)
        self.conv2 = nn.Sequential(self.encoder[5], self.relu, self.encoder[7], self.relu)
        self.conv3 = nn.Sequential(self.encoder[10], self.relu, self.encoder[12], self.relu, self.encoder[14], self.relu)
        self.conv4 = nn.Sequential(self.encoder[17], self.relu, self.encoder[19], self.relu, self.encoder[21], self.relu)
        self.conv5 = nn.Sequential(self.encoder[24], self.relu, self.encoder[26], self.relu, self.encoder[28], self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, class_num, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        return self.final(dec1)
