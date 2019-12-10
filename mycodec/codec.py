__author__ = 'SherlockLiao'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

from dataset import get_loader

train_dir="../../data/eval"
eval_dir="../../data/eval"
train_mv_dir="../../data/eval_mv"
eval_mv_dir="../../data/eval_mv"

train_loader = get_loader(
  is_train=True,
  root=train_dir, mv_dir=train_mv_dir,
)

def get_eval_loaders():
  eval_loaders = {
    'TVL': get_loader(
        is_train=False,
        root=eval_dir, mv_dir=eval_mv_dir),
  }
  return eval_loaders


if not os.path.exists('./output'):
    os.mkdir('./output')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    #x = x.view(x.size(0), 3, 352, 640)
    x = x.view(x.size(0), 3, 288, 352)
    print (x.shape)
    return x


num_epochs = 20
batch_size = 2
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        print(x.shape)
        x = self.encoder(x)
        print(x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.ConvTranspose2d(16, 3, 3)

    def forward(self, x):
        print(x.shape)
        x = self.decoder(x)
        print(x.shape)
        return x

def train_autoencoder(encoder, decoder):
    criterion = nn.MSELoss()
    nets = [encoder, decoder]
    params = [{'params': net.parameters()} for net in nets]
    optimizer = torch.optim.Adam(params, lr=learning_rate,
                                 weight_decay=1e-5)
    
    for epoch in range(num_epochs):
        itr = 0
        for img, fn in train_loader:
            img = Variable(img).cuda()
    
            # ===================forward=====================
            encoded_output = encoder(img)
            decoded_output = decoder(encoded_output)
    
            loss = criterion(decoded_output, img)
    
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            itr += 1
            print('Iteration:{}, loss:{:.4f}'.format(itr, loss.data))

def test_autoencoder(encoder, decoder):
    for img, fn in train_loader:
        img = Variable(img).cuda()
        code = encoder(img)
        output = decoder(code)
        pic = to_img(output.cpu().data)
        save_image(pic, './output/image_temp.png')

encoder = Encoder().cuda()
decoder = Decoder().cuda()

train_autoencoder(encoder, decoder)

torch.save(encoder.state_dict(), './models/encoder.pth')
torch.save(decoder.state_dict(), './models/decoder.pth')

test_autoencoder(encoder, decoder)
