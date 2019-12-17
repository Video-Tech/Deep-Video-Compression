from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, (1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(32, 64, (1, 3, 3), padding=(0, 1, 1))
        self.conv3 = nn.Conv3d(64, 128, (1, 3, 3), padding=(0, 1, 1))
        self.mp = nn.MaxPool3d((1, 2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.mp(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv3d(128, 64, (1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(64, 32, (1, 3, 3), padding=(0, 1, 1))
        self.conv3 = nn.Conv3d(32, 3, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
        x = self.conv3(x)
        return x
