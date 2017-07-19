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
        import time
        tic = time.time()
        self.em.Reload()
        logger.info('reload consume time {}'.format(time.time() - tic))

        tic = time.time()
        self.scalars_names = self.em.Runs()[self.name]['scalars']
        self.tensors_names = self.em.Runs()[self.name]['tensors']
        self.scalars = {}
        self.tensors = {}
        self.load_scalars()
        logger.info('load scalars consmue {}'.format(time.time() - tic))
        tic = time.time()
        self.load_tensors()
        logger.info('load tensors consume {}'.format(time.time() - tic))

    def load_scalars(self):
        for scalar_name in self.scalars_names:
            self.scalars[scalar_name] = [e.value for e in self.em.Scalars(self.name, scalar_name)]

    def load_tensors(self):
        for tensor_name in self.tensors_names:

            e = self.em.Tensors(self.name, tensor_name)[0]
            last_step = e.step
            tensors = [make_ndarray(e.tensor_proto)]
            for e in self.em.Tensors(self.name, tensor_name)[1:]:
                if last_step == e.step:
                    tensors[e.step] = np.concatenate((tensors[e.step], make_ndarray(e.tensor_proto)), axis=0)
                else:
                    tensors.append(make_ndarray(e.tensor_proto))
                last_step = e.step

            self.tensors[tensor_name] = tensors


if __name__ == '__main__':
    loader = Loader('vgg11_cifar10', Config.root_path + '/tfevents/vgg11_cifar10/..')
    print loader.scalars
    print [t for t in loader.tensors]
    print [len(t) for t in loader.tensors.values()]
    print [t[0].shape for t in loader.tensors.values()]
