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

train_dir="../data/train"
eval_dir="../data/eval1"
train_mv_dir="../data/train_mv"
eval_mv_dir="../data/eval1_mv"

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


if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 352, 640)
    return x


num_epochs = 20
batch_size = 100
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

encoder = Encoder().cuda()
decoder = Decoder().cuda()
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
        #output = model(img)
        code = encoder(img)
        output = decoder(code)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        itr += 1
        print('Iteration:{}, loss:{:.4f}'.format(itr, loss.data))
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(encoder.state_dict(), './conv_encoder.pth')
torch.save(decoder.state_dict(), './conv_decoder.pth')
