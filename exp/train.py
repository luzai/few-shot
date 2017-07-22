import os

os.environ['DISPLAY'] = 'localhost:11.0'
import matplotlib

matplotlib.use('TkAgg')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PATH'] = '/home/gyzhang/cuda-8.0/bin:' + os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = '/home/gyzhang/cuda-8.0/lib64'
import tensorflow as tf

tf_graph = tf.get_default_graph()
_sess_config = tf.ConfigProto(allow_soft_placement=True)
_sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=_sess_config, graph=tf_graph)
import keras.backend as K

K.set_session(sess)

import keras
from saver import TensorBoard
from datasets import Dataset
from models import VGG
from opts import Config

config = Config(epochs=11, batch_size=256, verbose=2,
                name='vgg11_cifar10',
                model_type='vgg11',
                dataset_type='cifar10',
                debug=True)

dataset = Dataset(config.dataset_type,debug=config.debug)
vgg = VGG(dataset.input_shape, dataset.classes,config.model_type, with_bn=False)
# todo lr scheme
# todo plataea auto detect and variant sample rate
# todo limit validation dataset size
vgg.model.compile(keras.optimizers.sgd(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
import tensorflow as tf

vgg.model.fit(dataset.x_train, dataset.y_train, batch_size=config.batch_size, epochs=config.epochs,
              verbose=config.verbose,
              validation_data=(dataset.x_test, dataset.y_test),
              callbacks=[TensorBoard(log_dir=config.model_tfevents_path,
                                     histogram_freq=5,
                                     batch_size=config.batch_size,
                                     write_graph=False,
                                     write_grads=False,
                                     )])


