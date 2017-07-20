from loader import Loader
from log import logger
from opts import Config
import time
import pandas as pd

loader = Loader('vgg11_cifar10', Config.root_path + '/tfevents/vgg11_cifar10/..')
tic = time.time()
scalars = loader.load_scalars()
logger.info('load scalars consmue {}'.format(time.time() - tic))
tic = time.time()
tensors = loader.load_tensors()
logger.info('load tensors consume {}'.format(time.time() - tic))

# analysor = pd.merge(scalars,tensors)

# todo clean name
