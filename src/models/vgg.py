import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPool2D, \
  BatchNormalization, Activation, Dropout
import tensorflow as tf
from models import BaseModel
from keras.regularizers import l2


class VGG(BaseModel):
  model_type = ['vgg11', 'vgg13', 'vgg16', 'vgg6', 'vgg19', 'vgg10', 'vgg9', 'vgg8']
  
  def __init__(self, input_shape, classes, config, with_bn=True, with_dp=True,hiddens=512):
    super(VGG, self).__init__(input_shape, classes, config, with_bn, with_dp,hiddens)
    type = config.model_type
    cfg = {
      'vgg11': [[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], [512, 512, self.classes]],
      'vgg13': [[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                [512, 512, self.classes]],
      'vgg16': [[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                [512, 512, self.classes]],
      'vgg19': [[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                 'M'], [512, 512, self.classes]],
      'vgg6' : [[32, 32, 'M', 64, 64, 'M'], [512, self.classes]],
      'vgg10': [[32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M'], [512, self.hiddens, self.classes]],
      'vgg9' : [[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'], [512, 512, self.classes]],
      'vgg8' : [[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'], [512, self.classes]],
    }
    # convert to my coding
    self.arch = [['conv2d', _config] if _config != 'M' else ['maxpooling2d'] for _config in cfg[type][0]]
    self.arch += [['flatten']]
    self.arch += [['dense', _config] for _config in cfg[type][1]]
    self.model = self.build(name=config.name)
    self.vis()
  
  def build(self, name):
    x = input = Input(self.input_shape)
    depth = 1
    for config in self.arch:
      if config[0] == 'conv2d':
        if not self.with_bn:
          x = Conv2D(config[1], (3, 3), padding='same', name='layer{}/conv'.format(depth),
                     kernel_regularizer=l2(1.e-4))(x)
          x = Activation('relu')(x)
        else:
          x = Conv2D(config[1], (3, 3), padding='same', name='layer{}/conv'.format(depth),
                     kernel_regularizer=l2(1.e-4))(x)
          x = BatchNormalization(axis=-1, name='layer{}/batchnormalization'.format(depth))(x)
          x = Activation('relu')(x)
        if self.with_dp:
          x = Dropout(.35)(x)
        depth += 1
      elif config[0] == 'maxpooling2d':
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
      elif config[0] == 'flatten':
        x = Flatten()(x)
      elif config[0] == 'dense' and config[1] != self.classes:
        x = Dense(config[1], name='layer{}/dense'.format(depth))(x)
        x = Activation('relu')(x)
        depth += 1
        if self.with_dp:
          x = Dropout(.5)(x)
      else:
        assert config[1] == self.classes, 'should be end'
        x = Dense(config[1], name='Layer{}/dense'.format(depth), use_bias=False)(x)
        x = Activation('softmax', name='Layer{}/softmax'.format(depth))(x)
    
    model = Model(input, x, name=name)
    return model


if __name__ == '__main__':
  import sys
  
  sys.path.append('/home/wangxinglu/prj/Perf_Pred/src')
  from configs import Config
  
  config = Config(epochs=301, batch_size=256, verbose=2,
                  model_type='vgg10',
                  dataset_type='cifar10',
                  debug=False, )
  
  model = VGG((32, 32, 3), 10, config)
  
  model.vis()
