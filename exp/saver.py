from keras.callbacks import Callback
import keras.backend as K
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
from log import logger


def clean_name(name):
    import re
    name = re.findall('([\w/]+)(?::\d+)?', name)[0]
    name = re.findall('([\w/]+)(?:_\d+)?', name)[0]
    return name


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
        weight_summ_l = []
        grad_summ_l = []
        act_summ_l = []
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                if 'conv2d' not in layer.name:
                    continue
                for weight in layer.weights:
                    # todo more clean way to name
                    weight_summ_l.append(tf.summary.tensor_summary(clean_name(weight.op.name), weight))
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)
                        grad_summ_l.append(tf.summary.tensor_summary('{}/grad'.format(clean_name(weight.op.name)),
                                                                     grads))

                if hasattr(layer, 'output'):
                    act_summ_l.append(tf.summary.tensor_summary('{}/act'.format(clean_name(layer.name)),
                                                                layer.output))

        self.act_summ = tf.summary.merge(act_summ_l)
        self.grad_summ = tf.summary.merge(grad_summ_l) if grad_summ_l != [] else None
        self.weight_summ = tf.summary.merge(weight_summ_l)
        self.merged = tf.summary.merge(act_summ_l + weight_summ_l + grad_summ_l)

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir + '/miscellany',
                                                self.sess.graph)  # will write graph and loss and so on
        else:
            self.writer = tf.summary.FileWriter(self.log_dir + '/miscellany')

    def new_writer(self, act_l, weight, epoch):
        for ind, act in enumerate(act_l):
            writer_act = tf.summary.FileWriter(self.log_dir + '/act' + '/' + str(epoch) + '/' + str(ind))
            writer_act.add_summary(act, global_step=epoch)
            writer_act.close()

        writer_weight = tf.summary.FileWriter(self.log_dir + '/param' + '/' + str(epoch))
        # param include weight and bias
        writer_weight.add_summary(weight, epoch)
        writer_weight.close()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        logger.info('Epoch {} end'.format(epoch))
        if self.validation_data and self.histogram_freq and epoch % self.histogram_freq == 0:
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
            act_summ_str_l = []
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
                if i == 0:
                    weight_summ_str, act_summ_str = self.sess.run([self.weight_summ, self.act_summ],
                                                                  feed_dict=feed_dict)
                else:
                    # the weight is same but grad and act is dependent on inputs minibatch
                    act_summ_str = self.sess.run([self.act_summ], feed_dict=feed_dict)[0]
                act_summ_str_l.append(act_summ_str)
                i += self.batch_size
            self.new_writer(act_summ_str_l, weight_summ_str, epoch)

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
