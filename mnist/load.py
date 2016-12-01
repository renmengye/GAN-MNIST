import sys
sys.path.append('..')

import gzip
import numpy as np
import os
from time import time
from collections import Counter
import random
from matplotlib import pyplot as plt

data_dir = '/home/mren/code/unseg/data/mnist'


def mnist():
  ff = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
  with gzip.open(ff) as fd:
    # loaded = np.fromfile(file=fd, dtype=np.uint8)
    loaded = np.fromstring(fd.read(), dtype=np.uint8)
  trX = loaded[16:].reshape((60000, 28 * 28)).astype(float)

  ff = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
  with gzip.open(ff) as fd:
    # loaded = np.fromfile(file=fd, dtype=np.uint8)
    loaded = np.fromstring(fd.read(), dtype=np.uint8)
  trY = loaded[8:].reshape((60000))

  ff = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
  with gzip.open(ff) as fd:
    # loaded = np.fromfile(file=fd, dtype=np.uint8)
    loaded = np.fromstring(fd.read(), dtype=np.uint8)
  teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)

  ff = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
  with gzip.open(ff) as fd:
    # loaded = np.fromfile(file=fd, dtype=np.uint8)
    loaded = np.fromstring(fd.read(), dtype=np.uint8)
  teY = loaded[8:].reshape((10000))

  trY = np.asarray(trY)
  teY = np.asarray(teY)

  return trX, teX, trY, teY


def mnist_with_valid_set():
  trX, teX, trY, teY = mnist()

  train_inds = np.arange(len(trX))
  np.random.shuffle(train_inds)
  trX = trX[train_inds]
  trY = trY[train_inds]
  #trX, trY = shuffle(trX, trY)
  vaX = trX[50000:]
  vaY = trY[50000:]
  trX = trX[:50000]
  trY = trY[:50000]

  return trX, vaX, teX, trY, vaY, teY
