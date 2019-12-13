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

from network import Encoder
from network import Decoder
from dataset import get_loader

train_dir="../../data/train"
test_dir="../../data/eval1"
train_mv_dir="../../data/train_mv"
test_mv_dir="../../data/eval1_mv"

train_loader = get_loader(
  is_train=True,
  root=train_dir, mv_dir=train_mv_dir,
)

test_loader = get_loader(
  is_train=False,
  root=test_dir, mv_dir=test_mv_dir,
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
    x = x.view(x.size(0), 3, 3, 352, 640)
    return x

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

num_epochs = 2
learning_rate = 1e-3

def train_autoencoder(encoder, decoder):
    criterion = nn.MSELoss()
    nets = [encoder, decoder]
    params = [{'params': net.parameters()} for net in nets]
    optimizer = torch.optim.Adam(params, lr=learning_rate,
                                 weight_decay=1e-5)
    
    for epoch in range(num_epochs):
        itr = 0
        for img, fn in train_loader:
            img = img.permute(0, 4, 2, 3, 1)
            img = Variable(img).cuda()
    
            # ===================forward=====================
            encoded_output = encoder(img)
            decoded_output = decoder(encoded_output)

            if itr % 500 == 0:
                torch.save(encoder.state_dict(), './models/encoder_{}_{}.pth'.format(epoch, itr))
                torch.save(decoder.state_dict(), './models/decoder_{}_{}.pth'.format(epoch, itr))

            loss = criterion(decoded_output, img)
    
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            itr += 1
            print('Iteration:{}, loss:{:.4f}'.format(itr, loss.data))

def test_autoencoder(encoder, decoder):
    for img, fn in test_loader:
        img = img.permute(0, 4, 2, 3, 1)
        print (img.shape)
        img = Variable(img).cuda()
        code = encoder(img)
        output = decoder(code)
        pic = to_img(output.cpu().data)
        save_image(pic[0], './output/image_temp.png')

encoder = Encoder().cuda()
decoder = Decoder().cuda()

train_autoencoder(encoder, decoder)

test_autoencoder(encoder, decoder)
