from __future__ import division

import six
from keras.models import Model
from keras.layers import (
  Input,
  Activation,
  Dense,
  Flatten
)
from keras.layers.convolutional import (
  Conv2D,
  MaxPooling2D,
  AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


def _bn_relu(input):
  """Helper to build a BN -> relu block
  """
  # global with_bn
  # if with_bn:
  norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
  # else:
  #   norm = input
  return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
  """Helper to build a conv -> BN -> relu block
  """
  filters = conv_params["filters"]
  kernel_size = conv_params["kernel_size"]
  strides = conv_params.setdefault("strides", (1, 1))
  kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
  padding = conv_params.setdefault("padding", "same")
  kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
  # obs=conv_params.get('obs',None)
  global layer
  if layer is not None:
    name = 'layer{}/conv'.format(layer)
  else:
    name = None
  
  def f(input):
    conv = Conv2D(filters=filters, kernel_size=kernel_size,
                  strides=strides, padding=padding,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer, name=name)(input)
    return _bn_relu(conv)
  
  return f


def _bn_relu_conv(**conv_params):
  """Helper to build a BN -> relu -> conv block.
  This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
  """
  filters = conv_params["filters"]
  kernel_size = conv_params["kernel_size"]
  strides = conv_params.setdefault("strides", (1, 1))
  kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
  padding = conv_params.setdefault("padding", "same")
  kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
  # obs=conv_params.get('obs',None)
  global layer
  
  if layer is not None:
    name = 'layer{}/conv'.format(layer)
  else:
    name = None
  
  def f(input):
    activation = _bn_relu(input)
    return Conv2D(filters=filters, kernel_size=kernel_size,
                  strides=strides, padding=padding,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer, name=name)(activation)
  
  return f


def _shortcut(input, residual):
  """Adds a shortcut between input and residual block and merges them with "sum"
  """
  # Expand channels of shortcut to match residual.
  # Stride appropriately to match residual (width, height)
  # Should be int if network architecture is correctly configured.
  input_shape = K.int_shape(input)
  residual_shape = K.int_shape(residual)
  stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
  stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
  equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]
  
  global layer
  shortcut = input
  # 1 X 1 conv if shape is different. Else identity.
  if stride_width > 1 or stride_height > 1 or not equal_channels:
    shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                      kernel_size=(1, 1),
                      strides=(stride_width, stride_height),
                      padding="valid",
                      kernel_initializer="he_normal",
                      kernel_regularizer=l2(0.0001), name='layer{}/conv-s'.format(layer))(input)
  
  return add([shortcut, residual], name='layer{}/add'.format(layer))


def _residual_block(block_function, filters, repetitions, is_first_layer=False, obs=None):
  """Builds a residual block with repeating bottleneck blocks.
  """
  
  def f(input):
    
    for i in range(repetitions):
      init_strides = (1, 1)
      if i == 0 and not is_first_layer:
        init_strides = (2, 2)
      input = block_function(filters=filters, init_strides=init_strides,
                             is_first_block_of_first_layer=(is_first_layer and i == 0),
                             obs=obs)(input)
    
    return input
  
  return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, obs=None):
  """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
  Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
  """
  
  def f(input):
    global layer
    if is_first_block_of_first_layer:
      # don't repeat bn->relu since we just did bn->relu->maxpool
      conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                     strides=init_strides,
                     padding="same",
                     kernel_initializer="he_normal",
                     kernel_regularizer=l2(1e-4), name='layer{}/conv'.format(layer))(input)
    
    else:
      conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                            strides=init_strides)(input)
    layer += 1
    residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
    res = _shortcut(input, residual)
    layer += 1
    return res
  
  return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
  """Bottleneck architecture for > 34 layer resnet.
  Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

  Returns:
      A final conv layer of filters * 4
  """
  
  def f(input):
    global layer
    if is_first_block_of_first_layer:
      # don't repeat bn->relu since we just did bn->relu->maxpool
      conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                        strides=init_strides,
                        padding="same",
                        kernel_initializer="he_normal",
                        kernel_regularizer=l2(1e-4))(input)
    else:
      conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                               strides=init_strides)(input)
    conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
    residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
    res = _shortcut(input, residual)
    layer += 1
    return res
  
  return f


def init_obs():
  global layer
  layer = 1


def _handle_dim_ordering():
  global ROW_AXIS
  global COL_AXIS
  global CHANNEL_AXIS
  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
  else:
    CHANNEL_AXIS = 1
    ROW_AXIS = 2
    COL_AXIS = 3


def _get_block(identifier):
  if isinstance(identifier, six.string_types):
    res = globals().get(identifier)
    if not res:
      raise ValueError('Invalid {}'.format(identifier))
    return res
  return identifier


class ResnetBuilder(object):
  @staticmethod
  def build(input_shape, num_outputs, block_fn, repetitions, name='model', hiddens=512):
    """Builds a custom ResNet like architecture.

    Args:
        input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
        num_outputs: The number of outputs at final softmax layer
        block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
            The original paper used basic_block for layers < 50
        repetitions: Number of repetitions of various block units.
            At each block unit, the number of filters are doubled and the input size is halved

    Returns:
        The keras `Model`.
    """
    init_obs()
    _handle_dim_ordering()
    if len(input_shape) != 3:
      raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")
    
    # Permute dimension order if necessary
    # if K.image_dim_ordering() == 'tf':
    #     input_shape = (input_shape[2], input_shape[1], input_shape[0])
    
    # Load function from str if needed.
    block_fn = _get_block(block_fn)
    
    input = Input(shape=input_shape)
    conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
    global layer
    layer += 1
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)
    
    block = pool1
    filters = 64
    for i, r in enumerate(repetitions):
      block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0), )(
          block)  # obs
      filters *= 2
      # obs+=1
    
    # Last activation
    block = _bn_relu(block)
    
    # Classifier block
    block_shape = K.int_shape(block)
    # pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
    #                          strides=(1, 1))(block)
    
    flatten1 = Flatten()(block)
    dense = Dense(units=512, kernel_initializer='he_normal',
                  name='layer{}/dense'.format(layer), activation='relu')(flatten1)
    layer += 1
    dense = Dense(units=hiddens, kernel_initializer='he_normal',
                  name='layer{}/dense'.format(layer), activation='relu')(dense)
    layer += 1
    dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                  name='Layer{}/dense'.format(layer),
                  use_bias=False)(dense)
    dense = Activation('softmax', name='Layer{}/softmax'.format(layer))(dense)
    
    model = Model(inputs=input, outputs=dense, name=name)
    return model
  
  @staticmethod
  def build_resnet_18(input_shape, num_outputs):
    return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])
  
  @staticmethod
  def build_resnet_34(input_shape, num_outputs):
    return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])
  
  @staticmethod
  def build_resnet_50(input_shape, num_outputs):
    return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])
  
  @staticmethod
  def build_resnet_101(input_shape, num_outputs):
    return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])
  
  @staticmethod
  def build_resnet_152(input_shape, num_outputs):
    return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])


from models import BaseModel


class ResNet(BaseModel):
  model_type = ['resnet5', 'resnet11', 'resnet32']
  
  def __init__(self, input_shape, classes, config, with_bn=True, with_dp=True, hiddens=512):
    super(ResNet, self).__init__(input_shape, classes, config, with_bn, with_dp, hiddens)
    type = config.model_type
    cfg = {
      'resnet6' : [1, 1],
      'resnet8' : [1, 1, 1],
      'resnet10': [1, 2, 1],
      'resnet12': [1, 2, 1, 1],
      'resnet18': [2, 2, 2, 2],
      'resnet34': [3, 4, 6, 3],
    }
    
    self.model = ResnetBuilder.build(input_shape, classes, basic_block, cfg[type], name=config.name,
                                     hiddens=self.hiddens)
    self.vis()


if __name__ == '__main__':
  import sys
  
  sys.path.append('/home/wangxinglu/prj/Perf_Pred/src')
  from configs import Config
  
  config = Config(epochs=301, batch_size=256, verbose=2,
                  model_type='resnet10',
                  dataset_type='cifar10',
                  debug=False, )
  
  model = ResNet((32, 32, 3), 10, config)
  
  model.vis()
