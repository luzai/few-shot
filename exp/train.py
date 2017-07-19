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
from keras.callbacks import TensorBoard
from datasets import Dataset
from models import VGG
from opts import Config

config = Config(epochs=10, batch_size=256, verbose=1, name='vgg11_cifar10',
                model_type='vgg11',
                dataset_type='cifar10')

dataset = Dataset(config.dataset_type)
vgg = VGG(dataset.input_shape, dataset.classes,config.model_type, with_bn=False)
# todo lr scheme
vgg.model.compile(keras.optimizers.sgd(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
import tensorflow as tf

vgg.model.fit(dataset.x_train, dataset.y_train, batch_size=config.batch_size, epochs=config.epochs,
              validation_data=(dataset.x_test, dataset.y_test),
              callbacks=[TensorBoard(log_dir=config.tf_model_path,
                                     histogram_freq=5,
                                     batch_size=256,
                                     write_graph=True,
                                     write_grads=True,
                                     # write_images=True,
                                     # embeddings_freq=2
                                     )])
# self.lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10,
#                                     min_lr=0.5e-7)
# self.early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5)
# def set_logger_path(self, name):
#     self.csv_logger = CSVLogger(osp.join(self.output_path, name))

