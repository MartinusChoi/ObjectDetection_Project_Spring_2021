from __future__ import division

import os
import argparse
import tqdm # state bar of 'for' loop
import random
import numpy as np

from PIL import Image # Python Imgae Library (use for image data)

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader # dataset을 sampler와 조합 => dataset을 순회할 수 있는 iterable을 만들어줌
# PyTorch Tensor의 Wrapper, Computational Graph에서 Node로 표현됨. => PyTorch Tensor와 동일한 API
# x => Variable
# x.data => 그 값을 가지는 Tensor
# x.grad => 어떤 스칼라 값에 대해 x의 변화도를 갖는 또 다른 Variable
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocater

