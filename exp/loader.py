from opts import Config
from tensorflow.tensorboard.backend.event_processing import event_accumulator
from tensorflow.tensorboard.backend.event_processing import event_multiplexer


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
        self.em.Reload()
        self.scalars_names = self.em.Runs()[self.name]['scalars']
        self.tensors_names=self.em.Runs()[self.name]['tensots']
        self.scalars={}
        self.tensors={}
        self.load_scalars()
        self.load_tensors()

    def load_scalars(self):
        for scalar_name in self.scalars_names:
            self.scalars[scalar_name] = self.em.Scalars(self.name,scalar_name)

    def load_tensors(self):
        for tensor_name in self.tensors_names:
            self.tensors[tensor_name] =self.em.Tensors(self.name,tensor_name)


if __name__ =='__main__':
    loader=Loader('vgg11_cifar10','tfevents/vgg11_cifar10')
    print loader.scalars