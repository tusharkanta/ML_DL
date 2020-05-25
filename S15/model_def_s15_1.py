import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class S11Model(nn.Module):
    def __init__(self):
        super(S11Model, self).__init__()

        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.Layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), bias=False, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.R1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.Layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), bias=False, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.Layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), bias=False, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.R2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), bias=False)
        )
        self.pool = nn.MaxPool2d(4, 2)

    def forward(self, x):
        x = self.preplayer(x)
        x = self.Layer1(x)
        r1 = self.R1(x)
        x = x + r1
        x = self.Layer2(x)
        x = self.Layer3(x)
        r2 = self.R2(x)
        x = x + r2
        x = self.pool(x)
        x = self.fc(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x)
        return x

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dcd1 = double_conv(6, 64)
        self.dcd2 = double_conv(64, 128)
        self.dcd3 = double_conv(128, 256)
        self.dcd4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dcu3 = double_conv(256 + 512, 256)
        self.dcu2 = double_conv(128 + 256, 128)
        self.dcu1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(1, 1), padding=0, bias=False)
        self.dcu3m = double_conv(256 + 512, 256)
        self.dcu2m = double_conv(128 + 256, 128)
        self.dcu1m = double_conv(128 + 64, 64)

        self.conv_lastm = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, x):
        # bgs = sample['bg']
        # bgfgs = sample['bg_fg']
        # c = torch.cat([bgs, bgfgs], dim=1)

        conv1 = self.dcd1(x)
        x1 = self.maxpool(conv1)

        conv2 = self.dcd2(x1)
        x2 = self.maxpool(conv2)

        conv3 = self.dcd3(x2)
        x3 = self.maxpool(conv3)

        x4 = self.dcd4(x3)

        x5 = self.upsample(x4)
        x6 = torch.cat([x5, conv3], dim=1)

        x7 = self.dcu3(x6)
        x8 = self.upsample(x7)
        x9 = torch.cat([x8, conv2], dim=1)

        x10 = self.dcu2(x9)
        x11 = self.upsample(x10)
        x12 = torch.cat([x11, conv1], dim=1)

        x12 = self.dcu1(x12)

        outD = self.conv_last(x12)

        x5M = self.upsample(x4)
        x6M = torch.cat([x5M, conv3], dim=1)

        x7M = self.dcu3(x6M)
        x8M = self.upsample(x7M)
        x9M = torch.cat([x8M, conv2], dim=1)

        x10M = self.dcu2(x9M)
        x11M = self.upsample(x10M)
        x12M = torch.cat([x11M, conv1], dim=1)

        x12M = self.dcu1(x12M)

        outM = self.conv_last(x12M)

        return outD, outM

base_model = models.resnet18(pretrained=False)

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=False)

        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


def doubleconv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
        nn.Conv2d(out_channels, out_channels, 1, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False)
    )


class Encoder(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Encoder, self).__init__()
        self.c1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv11 = nn.Conv2d(outchannels, outchannels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU()

        self.c2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv21 = nn.Conv2d(outchannels, outchannels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannels)

        self.dconv = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=1, stride=2, bias=False)

    def forward(self, x):
        idchannel = self.dconv(x)
        x1 = self.c1(x)
        x2 = self.conv11(x1)
        x3 = self.bn1(x2)
        x4 = self.relu(x3)
        x5 = self.c2(x4)
        x6 = self.conv21(x5)
        x7 = self.bn2(x6)
        x8 = x7
        x9 = self.relu(x8)
        return x9


class Decoder(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Decoder, self).__init__()
        self.c1 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, bias=False)
        self.conv11 = nn.Conv2d(outchannels, outchannels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU()
        self.c2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, bias=False)
        self.conv21 = nn.Conv2d(outchannels, outchannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannels)

        self.dconv = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        idchannel = self.dconv(x)
        x1 = self.c1(idchannel)
        x2 = self.conv11(x1)
        x3 = self.bn1(x2)
        x4 = self.relu(x3)
        x5 = self.c2(x4)
        x6 = self.conv21(x5)
        x7 = self.bn2(x6)
        x8 = self.relu(x7)
        return x8


class DepthMask(nn.Module):
    def __init__(self):
        super(DepthMask, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.fdm1 = Encoder(64, 128)
        self.fdm2 = Encoder(128, 256)
        self.fdm3 = Encoder(256, 512)
        self.fdm4 = Decoder(512, 256)
        self.fdm5 = Decoder(256, 128)
        self.fdm6 = Decoder(128, 64)
        self.fdm7 = Decoder(64, 64)
        self.fdm8 = Decoder(64, 32)
        self.lasconv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, bias=False)
        )
        self.fdm4v = Decoder(512, 256)
        self.fdm5v = Decoder(256, 128)
        self.fdm6v = Decoder(128, 64)
        self.fdm7v = Decoder(64, 64)
        self.fdm8v = Decoder(64, 32)
        self.lastconvd = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.c1(x)
        x1 = self.maxpool(x0)
        x2 = self.fdm1(x1)
        x3 = self.fdm2(x2)
        x4 = self.fdm3(x3)

        x4 = nn.functional.interpolate(x4, scale_factor=4, mode='bilinear')
        x5 = self.fdm4(x4)
        x5 += x3
        x5 = nn.functional.interpolate(x5, scale_factor=7, mode='bilinear')
        x6 = self.fdm5(x5)
        x2 = nn.functional.interpolate(x2, scale_factor=2, mode='bilinear')

        x6 += x2
        x6 = nn.functional.interpolate(x6, scale_factor=2, mode='bilinear')
        x7 = self.fdm6(x6)
        x7 += x1
        x7 = nn.functional.interpolate(x7, scale_factor=2, mode='bilinear')
        x8 = self.fdm7(x7)
        x8 += x0
        x8 = nn.functional.interpolate(x8, scale_factor=2, mode='bilinear')
        x9 = self.fdm8(x8)
        x10 = self.lasconv(x9)

        x5v = self.fdm4v(x4)
        x5v += x3
        x5v = nn.functional.interpolate(x5v, scale_factor=7, mode='bilinear')
        x6v = self.fdm5(x5v)
        x6v += x2
        x6v = nn.functional.interpolate(x6v, scale_factor=2, mode='bilinear')
        x7v = self.fdm6(x6v)
        x7v = nn.functional.interpolate(x7v, scale_factor=2, mode='bilinear')
        x8v = self.fdm7(x7v)
        x8v += x0
        x8v = nn.functional.interpolate(x8v, scale_factor=2, mode='bilinear')
        x9v = self.fdm8(x8v)
        out10Depth = self.lastconvd(x9v)
        return x10, out10Depth
