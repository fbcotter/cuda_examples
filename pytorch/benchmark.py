from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'torch', 'cpp', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('-f', '--features', type=int, default=32)
parser.add_argument('-r', '--runs', type=int, default=100)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()

if options.example == 'py':
    from soft_thresh_test import SoftShrink
elif options.example == 'torch':
    from torch.nn import Softshrink as SoftShrink
elif options.example == 'cpp':
    from soft_thresh_test import SoftShrinkC as SoftShrink
else:
    from soft_thresh_test import SoftShrinkCUDA as SoftShrink
    options.cuda = True


if options.cuda:
    assert torch.cuda.is_available()
    dev = torch.device('cuda')
    X = torch.randn(options.batch_size, options.features, 100, 100,
                    requires_grad=True, device=dev)
    S = SoftShrink(.5).cuda()
else:
    X = torch.randn(options.batch_size, options.features, 100, 100,
                    requires_grad=True)
    S = SoftShrink(.5)

# Force CUDA initialization
y = S(X)
y.backward(torch.ones_like(y))

forward_min = math.inf
forward_time = 0
backward_min = math.inf
backward_time = 0
for _ in range(options.runs):
    S.zero_grad()
    X.grad.zero_()

    start = time.time()
    y = S(X)
    elapsed = time.time() - start
    forward_min = min(forward_min, elapsed)
    forward_time += elapsed

    start = time.time()
    y.backward(torch.ones_like(y))
    elapsed = time.time() - start
    backward_min = min(backward_min, elapsed)
    backward_time += elapsed

scale = TIME_SCALES[options.scale]
forward_min *= scale
backward_min *= scale
forward_average = forward_time / options.runs * scale
backward_average = backward_time / options.runs * scale

print('Forward: {0:.3f}/{1:.3f} {4} | Backward {2:.3f}/{3:.3f} {4}'.format(
    forward_min, forward_average, backward_min, backward_average,
    options.scale))
