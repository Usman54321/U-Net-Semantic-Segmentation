import torch.nn as nn
import torch
import torchvision.models
import torch.nn.functional as F


class SumUNet(nn.Module):
    def __init__(self, n_class, load_pretrained_encoder_layers=False):
        super().__init__()

        # The number of channels is 3 since the input image is RGB
        self.n_channels = 3
        self.n_classes = n_class
        self.load_pretrained_encoder_layers = load_pretrained_encoder_layers

        # Load the pretrained Resnet18 model depending on the flag
        self.base_model = torchvision.models.resnet18(
            pretrained=self.load_pretrained_encoder_layers)
        self.base_layers = list(self.base_model.children())

        # We need to convolve the 3 channels to 64 channels
        self.first_resnet_layers = nn.Sequential(
            *self.base_layers[:4]
        )

        # Contracting part of the network
        self.encoder = UNetEncoder(self.base_layers)

        # Expanding part of the network
        self.decoder = SumUNetDecoder(n_class)

    def forward(self, x):
        # The input image is RGB, so we need to convolve it to 64 channels
        x = self.first_resnet_layers(x)
        x, skip_connection = self.encoder(x)
        logits = self.decoder(x, skip_connection)
        return logits


class SumUNetDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.up2 = SumConcatUpsample(512, 256)
        self.up3 = SumConcatUpsample(256, 128)
        self.up4 = SumConcatUpsample(128, 64)
        self.outc = nn.ConvTranspose2d(
            64, num_classes, kernel_size=4, stride=4)

    def forward(self, x, skip_connection):
        x = self.up2(x, skip_connection[2])
        x = self.up3(x, skip_connection[1])
        x = self.up4(x, skip_connection[0])

        logits = self.outc(x)
        return logits

class SumConcatUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up = nn.ConvTranspose2d(
            self.in_channels, self.out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConvolution(self.out_channels, self.out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        x = torch.add(x, skip_connection)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_class, load_pretrained_encoder_layers=False):
        super().__init__()

        # The number of channels is 3 since the input image is RGB
        self.n_channels = 3
        self.n_classes = n_class
        self.load_pretrained_encoder_layers = load_pretrained_encoder_layers

        # Load the pretrained Resnet18 model depending on the flag
        self.base_model = torchvision.models.resnet18(
            pretrained=self.load_pretrained_encoder_layers)
        self.base_layers = list(self.base_model.children())

        # We need to convolve the 3 channels to 64 channels
        self.first_resnet_layers = nn.Sequential(
            *self.base_layers[:4]
        )

        # Contracting part of the network
        self.encoder = UNetEncoder(self.base_layers)

        # Expanding part of the network
        self.decoder = UNetDecoder(n_class)

    def forward(self, x):
        # The input image is RGB, so we need to convolve it to 64 channels
        x = self.first_resnet_layers(x)
        x, skip_connection = self.encoder(x)
        logits = self.decoder(x, skip_connection)
        return logits


class UNetEncoder(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer

        # The encoder is the first four layers of Resnet18
        self.layer1 = self.base_layer[4]        # 64 channels
        self.layer2 = self.base_layer[5]        # 128 channels
        self.layer3 = self.base_layer[6]        # 256 channels
        self.layer4 = self.base_layer[7]        # 512 channels

    def forward(self, x):
        x = self.layer1(x)
        skip_connection2 = x            # 64 channels
        x = self.layer2(x)
        skip_connection3 = x            # 128 channels
        x = self.layer3(x)
        skip_connection4 = x            # 256 channels
        x = self.layer4(x)

        return x, [skip_connection2, skip_connection3, skip_connection4]


class UNetDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.up2 = ConcatUpsample(512, 256)
        self.up3 = ConcatUpsample(256, 128)
        self.up4 = ConcatUpsample(128, 64)
        self.outc = nn.ConvTranspose2d(
            64, num_classes, kernel_size=4, stride=4)

    def forward(self, x, skip_connection):
        x = self.up2(x, skip_connection[2])
        x = self.up3(x, skip_connection[1])
        x = self.up4(x, skip_connection[0])

        logits = self.outc(x)
        return logits


class ConcatUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up = nn.ConvTranspose2d(
            self.in_channels, self.out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConvolution(self.in_channels, self.out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv(x)
        return x


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
