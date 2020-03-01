import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from modules import ConvLSTMCell, Sign

import functools


class EncoderCell(nn.Module):
    def __init__(self, v_compress, stack, fuse_encoder, fuse_level):
        super(EncoderCell, self).__init__()

        # Init.
        self.v_compress = v_compress
        self.fuse_encoder = fuse_encoder
        self.fuse_level = fuse_level
        if fuse_encoder:
            print('\tEncoder fuse level: {}'.format(self.fuse_level))

        # Layers.
        self.conv = nn.Conv2d(
            9 if stack else 3, 
            64, 
            kernel_size=3, stride=2, padding=1, bias=False)

        self.rnn1 = ConvLSTMCell(
            64,#128 if fuse_encoder and v_compress else 64,
            128,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn2 = ConvLSTMCell(
            128,#((384 if fuse_encoder and v_compress else 256) 
             #if self.fuse_level >= 2 else 256),
            128,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn3 = ConvLSTMCell(
            128,#((384 if fuse_encoder and v_compress else 512) 
             #if self.fuse_level >= 3 else 256),
            128,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)


    def forward(self, input, hidden1, hidden2 ,hidden3):

        x = self.conv(input)
        
        # Fuse
        #if self.v_compress and self.fuse_encoder:
        #    x = torch.cat([x, unet_output1[2], unet_output2[2]], dim=1)
        
        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]
        
        # Fuse.
        #if self.v_compress and self.fuse_encoder and self.fuse_level >= 2:
        #    print("compress h1[0]")
        #    x = torch.cat([x, unet_output1[1], unet_output2[1]], dim=1)
        
        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]
        
        # Fuse.
        #if self.v_compress and self.fuse_encoder and self.fuse_level >= 3:
        #    x = torch.cat([x, unet_output1[0], unet_output2[0]], dim=1)
        
        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]
        return x, hidden1, hidden2, hidden3


class Binarizer(nn.Module):
    def __init__(self, bits):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, input):
        feat = self.conv(input)
        x = F.tanh(feat)
        return self.sign(x)


class DecoderCell(nn.Module):
    def __init__(self, v_compress, shrink, bits, fuse_level):

        super(DecoderCell, self).__init__()

        # Init.
        self.v_compress = v_compress
        self.fuse_level = fuse_level
        print('\tDecoder fuse level: {}'.format(self.fuse_level))

        # Layers.
        self.conv1 = nn.Conv2d(
            64, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.rnn1 = ConvLSTMCell(
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn2 = ConvLSTMCell(
            (((128 + 256 // shrink * 2) if v_compress else 128) 
                if self.fuse_level >= 3 else 32), #out1=256
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn3 = ConvLSTMCell(
            (((128 + 128//shrink*2) if v_compress else 128) 
                if self.fuse_level >= 2 else 32), #out2=128
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)

        self.rnn4 = ConvLSTMCell(
            32, #(64) if v_compress else 64, #out3=64
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)

        self.conv2 = nn.Conv2d(
            32,
            3, 
            kernel_size=1, stride=1, padding=0, bias=False)



    def forward(self, input, hidden1, hidden2, hidden3, hidden4):

        x = self.conv1(input)
        hidden1 = self.rnn1(x, hidden1)

        # rnn 2
        x = hidden1[0]
        x = F.pixel_shuffle(x, 2)

        #if self.v_compress and self.fuse_level >= 3:
        #    x = torch.cat([x, unet_output1[0], unet_output2[0]], dim=1)

        hidden2 = self.rnn2(x, hidden2)

        # rnn 3
        x = hidden2[0]
        x = F.pixel_shuffle(x, 2)

        #if self.v_compress and self.fuse_level >= 2:
        #    x = torch.cat([x, unet_output1[1], unet_output2[1]], dim=1)

        hidden3 = self.rnn3(x, hidden3)

        # rnn 4
        x = hidden3[0]
        x = F.pixel_shuffle(x, 2)

        #if self.v_compress:
        #    x = torch.cat([x, unet_output1[2], unet_output2[2]], dim=1)

        hidden4 = self.rnn4(x, hidden4)

        # final
        x = hidden4[0]
        x = F.pixel_shuffle(x, 2)

        x = F.tanh(self.conv2(x)) / 2
        return x, hidden1, hidden2, hidden3, hidden4


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    net.cuda()
    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_D(input_nc=6, ndf=64, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator
    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    # normalization layer
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    # define the main discriminator net
    net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)

    return init_net(net, init_type, init_gain)
