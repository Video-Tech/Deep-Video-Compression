from collections import namedtuple
from scipy.misc import imsave
import cv2
import numpy as np
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

def init_lstm(batch_size, height, width):

    encoder_h_1 = (Variable(
        torch.zeros(batch_size, 128, height // 4, width // 4)),
                   Variable(
                       torch.zeros(batch_size, 128, height // 4, width // 4)))
    encoder_h_2 = (Variable(
        torch.zeros(batch_size, 128, height // 8, width // 8)),
                   Variable(
                       torch.zeros(batch_size, 128, height // 8, width // 8)))
    encoder_h_3 = (Variable(
        torch.zeros(batch_size, 128, height // 16, width // 16)),
                   Variable(
                       torch.zeros(batch_size, 128, height // 16, width // 16)))

    decoder_h_1 = (Variable(
        torch.zeros(batch_size, 128, height // 16, width // 16)),
                   Variable(
                       torch.zeros(batch_size, 128, height // 16, width // 16)))
    decoder_h_2 = (Variable(
        torch.zeros(batch_size, 128, height // 8, width // 8)),
                   Variable(
                       torch.zeros(batch_size, 128, height // 8, width // 8)))
    decoder_h_3 = (Variable(
        torch.zeros(batch_size, 128, height // 4, width // 4)),
                   Variable(
                       torch.zeros(batch_size, 128, height // 4, width // 4)))
    decoder_h_4 = (Variable(
        torch.zeros(batch_size, 256 if False else 128, height // 2, width // 2)),
                   Variable(
                       torch.zeros(batch_size, 256 if False else 128, height // 2, width // 2)))

    encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
    encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
    encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

    decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
    decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
    decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
    decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

    return (encoder_h_1, encoder_h_2, encoder_h_3, 
            decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
