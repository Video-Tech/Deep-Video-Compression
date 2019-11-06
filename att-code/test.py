import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from torch.autograd import Variable

from dataset import get_loader
from evaluate import run_eval
from train_options import parser
from util import get_models, init_lstm, set_train, set_eval
from util import prepare_inputs, forward_ctx
from saliency.static_saliency import saliency_map

args = parser.parse_args()
print(args)

def get_eval_loaders():
  # We can extend this dict to evaluate on multiple datasets.
  eval_loaders = {
    'TVL': get_loader(
        is_train=False,
        root=args.eval, mv_dir=args.eval_mv,
        args=args),
  }
  return eval_loaders

############### Model ###############
encoder, binarizer, decoder, unet = get_models(
  args=args, v_compress=args.v_compress, 
  bits=args.bits,
  encoder_fuse_level=args.encoder_fuse_level,
  decoder_fuse_level=args.decoder_fuse_level)

nets = [encoder, binarizer, decoder, unet]

def load_model(index):
  names = ['encoder', 'binarizer', 'decoder', 'unet']

  for net_idx, net in enumerate(nets):
    if net is not None:
      name = names[net_idx]
      checkpoint_path = '{}/{}_{}_{:08d}.pth'.format(
          args.model_dir, args.save_model_name, 
          name, index)

      print('Loading %s from %s...' % (name, checkpoint_path))
      net.load_state_dict(torch.load(checkpoint_path))

load_model('gaze_model', 9000)

set_eval(nets)

eval_loaders = get_eval_loaders()
for eval_name, eval_loader in eval_loaders.items():
    eval_begin = time.time()
    eval_loss, mssim, psnr, att_msssim, att_psnr = run_eval(nets, eval_loader, args,
        output_suffix='iter%d' % train_iter)

    print('Evaluation @iter %d done in %d secs' % (
        train_iter, time.time() - eval_begin))
    print('%s Loss   : ' % eval_name
          + '\t'.join(['%.5f' % el for el in eval_loss.tolist()]))
    print('%s MS-SSIM: ' % eval_name
          + '\t'.join(['%.5f' % el for el in mssim.tolist()]))
    #print('%s ATT MS-SSIM: ' % eval_name
    #      + '\t'.join(['%.5f' % el for el in att_msssim.tolist()]))
    print('%s PSNR   : ' % eval_name
          + '\t'.join(['%.5f' % el for el in psnr.tolist()]))
    #print('%s ATT PSNR   : ' % eval_name
    #      + '\t'.join(['%.5f' % el for el in att_psnr.tolist()]))

