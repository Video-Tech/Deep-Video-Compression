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
from network import Binarizer
from network import Decoder
from dataset import get_loader
from options import parser
from train import train_codec
from test import test_codec

args = parser.parse_args()
print(args)

train_dir="../../data/train"
test_dir="../../data/eval1"
train_mv_dir="../../data/train_mv"
test_mv_dir="../../data/eval1_mv"

train_loader = get_loader(
  is_train=False,
  root=train_dir, mv_dir=train_mv_dir,
)

test_loader = get_loader(
  is_train=False,
  root=test_dir, mv_dir=test_mv_dir,
)

def load_model(encoder, decoder, epoch, itr):
    path = './models/encoder_{}_{}.pth'.format(epoch, itr)
    encoder.load_state_dict(torch.load(path))
    path = './models/decoder_{}_{}.pth'.format(epoch, itr)
    decoder.load_state_dict(torch.load(path))

def save_model(encoder, decoder, epoch, itr):
    torch.save(encoder.state_dict(), './models/encoder_{}_{}.pth'.format(epoch, itr))
    torch.save(decoder.state_dict(), './models/decoder_{}_{}.pth'.format(epoch, itr))

encoder = Encoder().cuda()
binarizer = Binarizer().cuda()
decoder = Decoder().cuda()

train_codec(train_loader, encoder, binarizer, decoder)

load_model(encoder, decoder, 1, 10000)

test_codec(test_loader, encoder, decoder)
