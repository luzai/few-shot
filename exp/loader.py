import time
import pandas as pd
from log import logger
from opts import Config
from tensorflow.tensorboard.backend.event_processing import event_accumulator
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
import tensorflow as tf
from tensorflow.contrib.util import make_ndarray
import numpy as np


class Loader:
    def __init__(self, name, model_tfevents_path):
        assert name in model_tfevents_path
        self.name = name
        self.model_tfevents_path = model_tfevents_path
        self.em = event_multiplexer.EventMultiplexer(
            size_guidance={event_accumulator.COMPRESSED_HISTOGRAMS: 1,
                           event_accumulator.IMAGES: 1,
                           event_accumulator.AUDIO: 1,
                           event_accumulator.SCALARS: 0,
                           event_accumulator.HISTOGRAMS: 1,
                           event_accumulator.TENSORS: 0}
        ).AddRunsFromDirectory(model_tfevents_path)
        tic = time.time()
        self.em.Reload()
        logger.info('reload consume time {}'.format(time.time() - tic))

        self.scalars_names = self.em.Runs()[self.name]['scalars']
        self.tensors_names = self.em.Runs()[self.name]['tensors']

    def load_scalars(self):
        scalars = {}
        for scalar_name in self.scalars_names:
            scalars[scalar_name] = [e.value for e in self.em.Scalars(self.name, scalar_name)]
        return pd.DataFrame.from_dict(scalars)

    def load_tensors(self):
        tensors = pd.DataFrame()
        for tensor_name in self.tensors_names:
            tensors_t = pd.Series(name=tensor_name)
            tensors_t.index = tensors_t.index.astype('int64')
            for e in self.em.Tensors(self.name, tensor_name):
                now_step = e.step
                now_tensor = make_ndarray(e.tensor_proto)
                # logger.info('step {} tensors {} shape {}'.format(now_step, tensor_name, now_tensor.shape))
                if now_step not in tensors_t:
                    tensors_t[now_step] = now_tensor
                else:
                    tensors_t[now_step] = np.concatenate((now_tensor, tensors_t[now_step]), axis=0)
            tensors = pd.concat((tensors, tensors_t), axis=1)
        # print tensors.index
        # from IPython import embed
        # embed()
        # todo tensor sort by name 
        return tensors

    def head_tensors(self):
        for tensor_name in self.tensors_names:
            for e in self.em.Tensors(self.name, tensor_name):
                now_step = e.step
                now_tensor = make_ndarray(e.tensor_proto)
                logger.info('step {} tensors {} shape {}'.format(now_step, tensor_name, now_tensor.shape))


if __name__ == '__main__':
    loader = Loader('vgg11_cifar10', Config.root_path + '/tfevents/vgg11_cifar10/..')
    tic = time.time()
    scalars = loader.load_scalars()
    logger.info('load scalars consmue {}'.format(time.time() - tic))
    tic = time.time()
    tensors = loader.load_tensors()
    logger.info('load tensors consume {}'.format(time.time() - tic))
    # loader.head_tensors()
    # print scalars
    # print tensors.index, tensors.columns
    for tensor_name, tensor_series in tensors.iteritems():
        print  tensor_name
        for ind, tensor_value in tensor_series.iteritems():
            print ind, tensor_value.shape
