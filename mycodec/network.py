from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, (1, 3, 3))
        self.conv2 = nn.Conv3d(16, 32, (1, 3, 3))
        self.conv3 = nn.Conv3d(32, 64, (1, 3, 3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose3d(64, 32, (1, 3, 3))
        self.conv2 = nn.ConvTranspose3d(32, 16, (1, 3, 3))
        self.conv3 = nn.ConvTranspose3d(16, 3, (1, 3, 3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
