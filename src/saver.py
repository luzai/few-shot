from keras.callbacks import Callback
import keras.backend as K
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os, numpy as np
from logs import logger
from stats import Stat
import utils


def clean_name(name):
    import re
    name = re.findall('([a-zA-Z0-9/]+)(?::\d+)?', name)[0]
    name = re.findall('([a-zA-Z0-9/]+)(?:_\d+)?', name)[0]
    return name


# todo save on the fly

# tensorboard2 is batch beased
class TensorBoard2(Callback):
    def __init__(self, log_dir='./logs',
                 histogram_freq=1,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 dataset=None,

                 batch_based=False,
                 stat_only=True):
        super(TensorBoard2, self).__init__()
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
        self.dataset = dataset
        self.iter_per_epoch = int(np.ceil(dataset.x_train.shape[0] / float(batch_size)))
        self.log_flag = True
        self.epoch = 0
        self.batch_based = batch_based
        self.stat_only = stat_only

    def set_model(self, model):
        self.name = model.name
        self.model = model
        self.sess = K.get_session()
        weight_summ_l = []
        grad_summ_l = []
        act_summ_l = []
        self.act_l = {}
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                if 'obs' not in layer.name:  # only log obs
                    continue
                for weight in layer.weights:
                    # todo more clean way to name
                    logger.info('summ log {}'.format(clean_name(weight.op.name)))
                    weight_summ_l.append(tf.summary.tensor_summary(clean_name(weight.op.name), weight))
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)
                        grad_summ_l.append(tf.summary.tensor_summary('{}/grad'.format(clean_name(weight.op.name)),
                                                                     grads))

                if hasattr(layer, 'output'):
                    act_summ_l.append(tf.summary.tensor_summary('{}/act'.format(clean_name(layer.name)),
                                                                layer.output))
                    self.act_l['{}/act'.format(clean_name(layer.name))] = layer.output

        self.act_summ = tf.summary.merge(act_summ_l) if act_summ_l != [] else None
        self.grad_summ = tf.summary.merge(grad_summ_l) if grad_summ_l != [] else None
        self.weight_summ = tf.summary.merge(weight_summ_l) if weight_summ_l != [] else None
        self.merged = tf.summary.merge(
            act_summ_l + weight_summ_l + grad_summ_l) if act_summ_l + weight_summ_l + grad_summ_l != [] else None

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

    def update_log_flag(self, logs):
        # todo lr scheme
        if self.iter < 30 * 200 and self.iter % 50 == 0:
            self.log_flag = True
        elif self.iter < 90 * 200 and self.iter % 100 == 0:
            self.log_flag = True
        elif self.iter > 90 * 200 and self.iter % 300 == 0:
            self.log_flag = True
        else:
            self.log_flag = False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch = epoch
        logger.info('Model {} Epoch {} end'.format(self.name, epoch))
        assert self.batch == self.iter_per_epoch - 1, 'should equal '

        if self.batch_based and self.validation_data and self.histogram_freq and epoch % self.histogram_freq == 0:
            logger.info('Epoch {} record tf merged summary'.format(epoch))
            act_summ_str_l, weight_summ_str = self.get_act_param_summ_str()
            self.new_writer(act_summ_str_l, weight_summ_str, epoch)

        if self.batch_based:
            self.write_single_value(logs, epoch)

    def on_batch_end(self, batch, logs=None):
        self.batch = batch
        self.iter = iter = self.epoch * self.iter_per_epoch + self.batch
        self.update_log_flag(logs)
        if self.validation_data and self.histogram_freq and self.log_flag:
            logger.info('Epoch {} Batch {} Iter {} end'.format(self.epoch, self.batch, iter))
            if not self.stat_only:
                act_summ_str_l, weight_summ_str = self.get_act_param_summ_str()
                self.new_writer(act_summ_str_l, weight_summ_str, iter)
            else:
                dl = []
                act = self.get_act()
                stat = Stat()
                for name, val in act.iteritems():
                    dl.append(stat.calc_all(val, name))
                param = self.get_param()
                for name, val in param.iteritems():
                    dl.append(stat.calc_all(val, name))
                d = utils.dict_concat(dl)

            val_loss, val_acc = self.model.evaluate(self.dataset.x_test, self.dataset.y_test, verbose=2)

            logs['val_loss'] = val_loss
            logs['val_acc'] = val_acc
            logs = utils.dict_concat([logs, d])
            self.write_single_value(logs, iter)

    def on_train_end(self, logs=None):
        self.writer.close()

    def get_param(self):
        res = {}
        for layer in self.model.layers:
            if 'conv2d' not in layer.name:
                continue
            kernel, bias = layer.get_weights()
            res[clean_name(layer.name) + '/kernel'] = kernel
            res[clean_name(layer.name) + '/bias'] = bias
        return res

    def get_act(self):
        val_data = self.validation_data
        tensors = (self.model.inputs +
                   self.model.targets +
                   self.model.sample_weights)

        if self.model.uses_learning_phase:
            tensors += [K.learning_phase()]

        assert len(val_data) == len(tensors)
        val_size = val_data[0].shape[0]
        i = 0
        res = {}
        while i < val_size:
            step = min(self.batch_size, val_size - i)
            # logger.info('Val size {} Now {} step forward {}'.format(val_size, i, step))
            batch_val = []
            batch_val.append(val_data[0][i:i + step])
            batch_val.append(val_data[1][i:i + step])
            batch_val.append(val_data[2][i:i + step])
            if self.model.uses_learning_phase:
                batch_val.append(np.zeros_like(val_data[3]))  # do not use dropout!
            feed_dict = dict(zip(tensors, batch_val))

            for _act_name, _act in self.sess.run(self.act_l, feed_dict=feed_dict).items():
                if _act_name in res:
                    res[_act_name] = np.concatenate((res[_act_name], _act), axis=0)
                else:
                    res[_act_name] = _act
            i += self.batch_size
        return res

    def get_act_param_summ_str(self):
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
            # logger.info('Val size {} Now {} step forward {}'.format(val_size, i, step))
            batch_val = []
            batch_val.append(val_data[0][i:i + step])
            batch_val.append(val_data[1][i:i + step])
            batch_val.append(val_data[2][i:i + step])
            if self.model.uses_learning_phase:
                batch_val.append(np.zeros_like(val_data[3]))
            feed_dict = dict(zip(tensors, batch_val))
            if i == 0:
                weight_summ_str, act_summ_str = self.sess.run([self.weight_summ, self.act_summ],
                                                              feed_dict=feed_dict)
                res = {}
                for _act_name, _act in self.sess.run(self.act_l, feed_dict=feed_dict).items():
                    res[_act_name] = _act

            # todo I do not write grad in fact
            else:
                # the weight is same but grad and act is dependent on inputs minibatch
                act_summ_str = self.sess.run([self.act_summ], feed_dict=feed_dict)[0]
            act_summ_str_l.append(act_summ_str)
            i += self.batch_size
        return act_summ_str_l, weight_summ_str

    def write_single_value(self, logs, epoch_iter):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, epoch_iter)


