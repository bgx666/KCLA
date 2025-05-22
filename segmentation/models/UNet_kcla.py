import torch
import torch.nn as nn
import torch.nn.functional as F
from kcla.kcla_forU import resnet18_kcla


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)  
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet_kcla(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_kcla, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.encoder = resnet18_kcla(num_classes=n_classes)

        # self.up1 = Up(1024, 256, bilinear)
        # self.up2 = Up(512, 128, bilinear)
        # self.up3 = Up(256, 64, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

        self.up1 = Up(256, 64, bilinear)
        self.up2 = Up(128, 32, bilinear)
        self.up3 = Up(64, 16, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)

        x = self.up1(x5, x4)  # x shape: [batch_size, 256, H/8, W/8]
        x = self.up2(x, x3)  # x shape: [batch_size, 128, H/4, W/4]
        x = self.up3(x, x2)  # x shape: [batch_size, 64, H/2, W/2]
        x = self.up4(x, x1)  # x shape: [batch_size, 64, H, W]
        logits = self.outc(x)  # logits shape: [batch_size, n_classes, H, W]

        return  None,torch.sigmoid(logits)

if __name__ == '__main__':
    net = UNet_kcla(n_channels=3, n_classes=1).cuda()
    from thop import profile
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(net, (dummy_input,))
    print('flops: %.2f M, params: %.2f k' % (flops / 1000000, params / 1000))
    print('net total parameters:', sum(param.numel() for param in net.parameters()))