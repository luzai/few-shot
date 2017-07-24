import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPool2D, \
    BatchNormalization, Activation, Dropout
import tensorflow as tf


class VGG:
    def __init__(self, input_shape, classes, type='vgg11', with_bn=True, with_dp=True):
        self.input_shape = input_shape
        self.with_bn = with_bn
        self.with_dp = with_dp
        self.classes = classes
        cfg = {
            'vgg11': [[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], [512, 512, self.classes]],
            'vgg13': [[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                      [512, 512, self.classes]],
            'vgg16': [[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                      [512, 512, self.classes]],
            'vgg19': [[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                       'M'], [512, 512, self.classes]],
            'vgg5': [[32, 32, 'M', 64, 64, 'M'], [512, self.classes]],
        }
        # convert to my coding
        self.arch = [['conv2d', config] if config != 'M' else ['maxpooling2d'] for config in cfg[type][0]]
        self.arch += [['flatten']]
        self.arch += [['dense', config] for config in cfg[type][1]]
        self.model = self.build()

    def build(self):
        x = input = Input(self.input_shape)
        depth = 0
        for config in self.arch:
            if config[0] == 'conv2d':
                if not self.with_bn:
                    x = Conv2D(config[1], (3, 3), padding='same', name='obs{}/conv2d'.format(depth))(x)
                    x = Activation('relu')(x)
                else:
                    x = Conv2D(config[1], (3, 3), padding='same', name='obs{}/conv2d'.format(depth))(x)
                    x = BatchNormalization(axis=-1)(x)
                    x = Activation('relu')(x)
                depth += 1
            elif config[0] == 'maxpooling2d':
                x = MaxPooling2D((2, 2), strides=(2, 2))(x)
                if self.with_dp:
                    x = Dropout(.25)(x)
            elif config[0] == 'flatten':
                x = Flatten()(x)
            elif config[0] == 'dense' and config[1] != self.classes:
                x = Dense(config[1], activation='relu')(x)
                if self.with_dp:
                    x = Dropout(.5)(x)
            else:
                assert config[1] == self.classes, 'should be end'
                x = Dense(config[1], activation='softmax')(x)

        model = Model(input, x)
        return model


if __name__ == '__main__':
    vgg = VGG((32, 32, 3), 10, type='vgg5')
    vgg.model.summary()
