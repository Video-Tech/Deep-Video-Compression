"""Training options."""

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--bits', default=64, type=int, 
                    help='Bottleneck code size.')

parser.add_argument('--max-train-iters', type=int, default=100000,
                    help='Max training iterations.')
parser.add_argument('--batch-size', type=int, default=1, 
                    help='Batch size.')
parser.add_argument('--test-batch-size', type=int, default=1,
                    help='Batch size for testing.')

parser.add_argument('--out-dir', type=str, default='output',
                    help='Output directory')
parser.add_argument('--model-dir', type=str, default='model',
                    help='Path to model folder.')
parser.add_argument('--load-model-name', type=str,
                    help='Checkpoint name to load. (Do nothing if not specified.)')
parser.add_argument('--load-iter', type=int,
                    help='Iteraction of checkpoint to load.')
parser.add_argument('--save-model-name', type=str, default='model',
                    help='Checkpoint name to save.')
parser.add_argument('--save-codes', action='store_true',
                    help='If true, write compressed codes during eval.')
parser.add_argument('--save-out-img', action='store_true',
                    help='If true, save output images during eval.')
parser.add_argument('--checkpoint-iters', type=int, default=1000,
                    help='Model checkpoint period.')
parser.add_argument('--eval-iters', type=int, default=2000,
                    help='Evaluation period.')
