from keras.callbacks import Callback
import keras.backend as K
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
from log import logger


class TensorBoard(Callback):
    def __init__(self, log_dir='./logs',
                 histogram_freq=1,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()

        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                if 'conv2d' not in layer.name:
                    continue
                for weight in layer.weights:
                    # todo more clean way to name
                    tf.summary.tensor_summary(weight.op.name.strip(':0'), weight)
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)
                        tf.summary.tensor_summary('{}_grad'.format(weight.name.strip(':0')), grads)

                if hasattr(layer, 'output'):
                    tf.summary.tensor_summary('{}_out'.format(layer.name),
                                              layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # todo full batch size?
        logger.info('Epoch {} end'.format(epoch))
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                logger.info('Epoch {} record tf merged summary'.format(epoch))
                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    logger.info('Val size {} Now {} step forward {}'.format(val_size, i, step))
                    batch_val = []
                    batch_val.append(val_data[0][i:i + step])
                    batch_val.append(val_data[1][i:i + step])
                    batch_val.append(val_data[2][i:i + step])
                    if self.model.uses_learning_phase:
                        batch_val.append(val_data[3])
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    # todo this step can be asynchronous
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, logs=None):
        self.writer.close()
