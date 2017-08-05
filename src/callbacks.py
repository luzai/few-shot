from keras.callbacks import Callback
import keras.backend as K
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os, numpy as np
from logs import logger
from stats import KernelStat, ActStat, BiasStat
from utils import clean_name
import utils, math


# todo save on the fly

# tensorboard2 is batch beased
class TensorBoard2(Callback):
    def __init__(self,
                 tot_epochs,
                 log_dir='./logs',
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 dataset=None,
                 batch_based=False,
                 stat_only=True,
                 max_win_size=33,
                 ):
        super(TensorBoard2, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.epochs = tot_epochs
        self.log_dir = log_dir
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.batch_size = batch_size
        self.dataset = dataset
        self.max_win_zise = max_win_size

        self.merged = None
        self.iter_per_epoch = int(np.ceil(dataset.x_train.shape[0] / float(batch_size)))
        self.epoch, self.iter = 0, 0
        self.batch_based = batch_based
        self.stat_only = stat_only

        log_freq = np.array([4, 2, 1])
        log_freq_bin = np.array([0, self.epochs // 10, self.epochs // 3, self.epochs])

        log_freq_bin_iter = log_freq_bin * self.iter_per_epoch
        # log_freq_bin_iter[0] = self.max_win_zise//2
        log_num = np.diff(log_freq_bin) * log_freq
        log_pnt = np.concatenate([
            np.linspace(start, stop, num, endpoint=False).astype(int)
            for num, start, stop in
            zip(log_num, log_freq_bin_iter[:-1], log_freq_bin_iter[1:])
        ])
        log_pnts_l = []
        for pnt in log_pnt:
            _arr = np.arange(pnt - int(self.max_win_zise // 2),
                             pnt + int(self.max_win_zise // 2) + 1).astype(int)
            if (_arr < 0).any():
                _arr += - _arr.min()
            log_pnts_l.append(_arr)
        self.log_pnts =log_pnts = np.concatenate(log_pnts_l)

        self.kernel_stat = KernelStat(self.max_win_zise, log_pnt)
        self.bias_stat = BiasStat(self.max_win_zise, log_pnt)
        self.act_stat = ActStat(self.max_win_zise, log_pnt)

    def set_model(self, model):
        self.name = model.name
        self.model = model
        self.sess = K.get_session()
        weight_summ_l = []
        grad_summ_l = []
        act_summ_l = []
        self.act_l = {}
        if self.merged is None:
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
        # todo use cleaned dir
        for ind, act in enumerate(act_l):
            writer_act = tf.summary.FileWriter(self.log_dir + '/act' + '/' + str(epoch) + '/' + str(ind))
            writer_act.add_summary(act, global_step=epoch)
            writer_act.close()

        writer_weight = tf.summary.FileWriter(self.log_dir + '/param' + '/' + str(epoch))
        # param include kernels(weights) and bias
        writer_weight.add_summary(weight, epoch)
        writer_weight.close()

    def judge_log(self, logs):
        # todo sample scheme

        if self.iter in self.log_pnts:
            return True
        else:
            return False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch = epoch
        logger.info('Model {} Epoch {} end'.format(self.name, epoch))
        assert self.batch == self.iter_per_epoch - 1, 'should equal '

        if self.batch_based and self.validation_data:
            logger.info('Epoch {} record tf merged summary'.format(epoch))
            act_summ_str_l, weight_summ_str = self.get_act_param_summ_str()
            self.new_writer(act_summ_str_l, weight_summ_str, epoch)

        if self.batch_based:
            self.write_dict(logs, epoch)

    def on_batch_end(self, batch, logs=None):
        self.batch = batch
        iter = self.epoch * self.iter_per_epoch + self.batch
        assert self.iter == iter, 'should same'
        self.iter = iter

        if self.validation_data and self.judge_log(logs):
            logger.info('Epoch {} Batch {} Iter {} end'.format(self.epoch, self.batch, iter))
            if not self.stat_only:
                act_summ_str_l, weight_summ_str = self.get_act_param_summ_str()
                self.new_writer(act_summ_str_l, weight_summ_str, iter)
            else:
                act = self.get_act()
                for name, val in act.iteritems():
                    self.write_df(self.act_stat.calc_all(val, name, iter))
                kernel, bias = self.get_param()
                for name, val in kernel.iteritems():
                    self.write_df(self.kernel_stat.calc_all(val, name, iter))
                for name, val in bias.iteritems():
                    self.write_df(self.bias_stat.calc_all(val, name, iter))

            val_loss, val_acc = self.model.evaluate(self.dataset.x_test, self.dataset.y_test, verbose=2)

            logs['val_loss'] = val_loss
            logs['val_acc'] = val_acc
            self.write_dict(logs, iter)

    def on_train_end(self, logs=None):
        self.writer.close()

    def get_param(self):
        res_kernel, res_bias = {}, {}
        for layer in self.model.layers:
            if 'conv2d' not in layer.name:
                continue
            kernel, bias = layer.get_weights()
            res_kernel[clean_name(layer.name) + '/kernel'] = kernel
            res_bias[clean_name(layer.name) + '/bias'] = bias
        return res_kernel, res_bias

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

    def write_df(self, df):
        for name, series in df.iteritems():
            for _iter, val in series.iteritems():
                if not math.isnan(val):
                    self.write_single_val(val, _iter, name)

    def write_dict(self, logs, epoch_iter):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            self.write_single_val(value, epoch_iter, name)

    def write_single_val(self, value, epoch_iter, name):
        summary = tf.Summary()
        summary_value = summary.value.add()

        summary_value.simple_value = value
        summary_value.tag = name
        self.writer.add_summary(summary, epoch_iter)
