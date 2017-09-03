from keras.callbacks import Callback
import keras.backend as K
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os, numpy as np, pandas as pd
from logs import logger
from stats import KernelStat, ActStat, BiasStat
from utils import clean_name

import utils, math, itertools, os.path as osp


# todo use hdf5 to save tensors
# tensorboard2 is batch based
class TensorBoard2(Callback):
    def __init__(self,
                 tot_epochs,
                 log_dir,
                 max_win_size,
                 batch_size,
                 write_graph,
                 write_grads,
                 dataset,
                 batch_based,
                 stat_only,  # True -- save on the fly
                 ):
        super(TensorBoard2, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.timer = utils.Timer()
        self.epochs = tot_epochs
        self.log_dir = log_dir
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.batch_size = batch_size
        self.dataset = dataset
        self.max_win_zise = max_win_size

        self.merged = None
        self.iter_per_epoch = int(np.ceil(dataset.x_train.shape[0] / float(batch_size)))
        self.iters = self.iter_per_epoch * self.epochs
        self.epoch, self.iter = 0, -1
        self.batch_based = batch_based
        self.stat_only = stat_only

        series = pd.Series(data=3,
                           index=(np.arange(start=0,
                                            stop=utils.get_config('epochs'),
                                            step=1. / utils.get_config('sample_rate')) *
                                  self.iter_per_epoch).astype(np.int64)
                           )
        # print series
        series1 = pd.Series()
        for (ind0, _), (ind1, _) in zip(series.iloc[:-1].iteritems(), series.iloc[1:].iteritems()):
            if ind0 < utils.get_config('sub_sample')[0] * self.iter_per_epoch:
                sample_rate = utils.get_config('sub_sample_rate')[0]
            elif ind0 < utils.get_config('sub_sample')[1] * self.iter_per_epoch:
                sample_rate = utils.get_config('sub_sample_rate')[1]
            else:
                sample_rate = utils.get_config('sub_sample_rate')[2]

            series1 = series1.append(
                pd.Series(data=2,
                          index=np.linspace(ind0, ind1, sample_rate, endpoint=False)[1:].astype(int))
            )

        series = series.append(series1).sort_index()

        series1 = pd.Series()
        for ind, _ in series.iteritems():
            series1 = series1.append([
                pd.Series(data=1, index=np.arange(ind - self.max_win_zise // 2, ind)),
                pd.Series(data=1, index=np.arange(ind + 1, ind + self.max_win_zise // 2 + 1)),
            ])
        log_pnts = series.append(series1).sort_index()
        log_pnts.index = - log_pnts.index.min() + log_pnts.index
        # utils.pickle(log_pnts, './log.pkl')
        if not np.array_equal(np.array(log_pnts.index), np.unique(log_pnts.index)):
            logger.info(' ok !! alias in sample')
        log_pnts = log_pnts.groupby(log_pnts.index).max()

        self.log_pnts = log_pnts

        self.kernel_stat = KernelStat(self.max_win_zise, log_pnts)
        self.bias_stat = BiasStat(self.max_win_zise, log_pnts)
        self.act_stat = ActStat(self.max_win_zise, log_pnts)

    def set_model(self, model):
        self.name = model.name
        self.model = model
        self.sess = K.get_session()
        weight_summ_l = []
        act_summ_l = []
        self.act_l = {}
        if self.merged is None:
            for layer in self.model.layers:
                if (not utils.get_config('last_only') and (layer.name.startswith('Layer') or layer.name.startswith('layer'))) or (utils.get_config('last_only') and layer.name.startswith('Layer')):
                    if hasattr(layer, 'output'):
                        self.act_l['{}/act'.format(clean_name(layer.name))] = layer.output
                        # add input temporarily to observe!
                        # self.act_l['{}/in'.format(clean_name(layer.name))]=layer.input
                if layer.name.startswith('Layer'):
                    for weight in layer.weights:
                        logger.info('summ log {}'.format(clean_name(weight.op.name)))
                        weight_summ_l.append(tf.summary.tensor_summary(clean_name(weight.op.name), weight))
                    if hasattr(layer, 'output'):
                        act_summ_l.append(tf.summary.tensor_summary('{}/act'.format(clean_name(layer.name)),
                                                                    layer.output))
                        act_summ_l.append(tf.summary.tensor_summary('{}/in'.format(clean_name(layer.name)),
                                                                    layer.input))
        self.act_summ = tf.summary.merge(act_summ_l) if act_summ_l != [] else None
        self.weight_summ = tf.summary.merge(weight_summ_l) if weight_summ_l != [] else None
        self.merged = tf.summary.merge(act_summ_l + weight_summ_l) if act_summ_l + weight_summ_l != [] else None
        self.writer = tf.summary.FileWriter(self.log_dir + '/miscellany')

    def new_writer(self, act_l, weight=None, epoch=None):
        # todo use cleaned dir
        for ind, act in enumerate(act_l):
            writer_act = tf.summary.FileWriter(self.log_dir + '/act' + '/' + str(epoch) + '/' + str(ind))
            writer_act.add_summary(act, global_step=epoch)
            writer_act.close()

        writer_weight = tf.summary.FileWriter(self.log_dir + '/param' + '/' + str(epoch))
        # param include kernels(weights) and bias
        writer_weight.add_summary(weight, epoch)
        writer_weight.close()

    def new_writer_s(self,s_summ,epoch):
        self.writer.add_summary(s_summ,epoch)

    def judge_log(self, logs):
        # todo adapative sample scheme
        if self.iter in self.log_pnts:
            return True
        else:
            return False

    def on_epoch_begin(self, epoch, logs=None):
        logger.info('Model {} Epoch {} begin'.format(self.name, epoch))
        self.epoch = epoch
        if self.iter == -1: return
        lr = K.get_value(self.model.optimizer.lr)
        logger.info('lr is ' + str(lr))
        lr_dict = {}
        lr_dict['lr'] = lr
        self.write_dict(lr_dict, self.iter)

    def on_epoch_end(self, epoch, logs=None):
        logger.info('Model {} Epoch {} end'.format(self.name, epoch))
        assert self.batch == self.iter_per_epoch - 1, 'should equal '

    @utils.timeit('batch end log stats consume')
    def on_batch_end(self, batch, logs=None):
        self.batch = batch
        iter = self.epoch * self.iter_per_epoch + self.batch
        self.iter += 1

        assert self.iter == iter, 'epoch {} batch {} iter {} selgiter  {}'.format(self.epochs, batch, iter,
                                                                                  self.iter)
        if self.validation_data and self.judge_log(logs):
            logger.debug('Epoch {} Batch {} Iter {} end'.format(self.epoch, self.batch, iter))

            if utils.get_config('log_stat'):
                act = self.get_act()
                for name, val in act.iteritems():
                    self.write_df(self.act_stat.calc_all(val, name, iter))

                kernel, bias = self.get_param()
                for name, val in kernel.iteritems():
                    self.write_df(self.kernel_stat.calc_all(val, name, iter))
                for name, val in bias.iteritems():
                    self.write_df(self.bias_stat.calc_all(val, name, iter))

            if self.iter >= self.log_pnts.index[-1]:
                # stdtime_tensor = {}
                # for name, val in act.iteritems():
                #   stdtime_tensor[(iter, name)] = self.act_stat.stdtime_inst.last_std[name]
                # for name, val in kernel.iteritems():
                #   stdtime_tensor[(iter, name)] = self.kernel_stat.stdtime_inst.last_std[name]
                # for name, val in bias.iteritems():
                #   stdtime_tensor[(iter, name)] = self.bias_stat.stdtime_inst.last_std[name]
                # utils.write_df(vis_utils.dict2df(stdtime_tensor), self.log_dir + '/stdtime.h5')

                # utils.pickle(self.act_stat.stdtime_inst.record, '../act_cache.pkl')
                # utils.pickle(self.kernel_stat.stdtime_inst.record, '../kernel_cache.pkl')
                # utils.pickle(self.bias_stat.stdtime_inst.record, '../bias_cache.pkl')
                #
                # example_df = self.act_stat.stdtime_inst.df.join(
                #     [self.kernel_stat.stdtime_inst.df, self.bias_stat.stdtime_inst.df],
                #     how='outer')
                # self.write_df(example_df)
                pass

            if self.log_pnts[self.iter] >= 2:
                val_loss, val_acc = self.model.evaluate(self.dataset.x_test, self.dataset.y_test, verbose=2)
                logs['val_loss'] = val_loss
                logs['val_acc'] = val_acc
                self.write_dict(logs, iter)

            if utils.get_config('log_tensor') and self.log_pnts[self.iter] >= 2:
                act_summ_str_l, weight_summ_str = self.get_act_param_summ_str()
                self.new_writer(act_summ_str_l, weight_summ_str, iter)

            s_summ_str = self.get_s_summ()
            self.new_writer_s(s_summ_str, iter)

    def get_s_summ(self):
        val_data = [self.dataset.x_test_ref, self.dataset.y_test_ref, np.ones_like(self.validation_data[2]),
                    np.zeros_like(self.validation_data[3])]
        # val_data = self.validation_data
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
            s_summ = tf.get_collection("s_summ")
            print tf.get_default_graph()
            s_summ_str = self.sess.run([s_summ],
                                       feed_dict=feed_dict)
            return s_summ_str

    def on_train_end(self, logs=None):
        self.writer.close()

    def get_param(self):
        # when to use: for calculate statistic information
        res_kernel, res_bias = {}, {}
        for layer in self.model.layers:
            if (not utils.get_config('last_only') and (
                        layer.name.startswith('Layer') or layer.name.startswith('layer'))) or (
                        utils.get_config('last_only') and layer.name.startswith('Layer')):
                # if layer.name.startswith('layer') or layer.name.startswith('Layer') :
                logger.debug(layer.name)
                weights = layer.get_weights()
                # for _weight in weights:
                #     print _weight.shape
                if weights == []: continue
                if len(weights) == 2:
                    kernel, bias = weights
                    res_kernel[clean_name(layer.name) + '/kernel'] = kernel
                    res_bias[clean_name(layer.name) + '/bias'] = bias
                elif len(weights) == 4:
                    assert 'gamma' in layer.weights[0].name
                    # gamma, beta, movingmean, movingvariance = weights
                    # res_kernel[clean_name(layer.name) + '/gamma'] = gamma
                    # res_bias[clean_name(layer.name) + '/beta'] = beta
                    # res_bias[clean_name(layer.name) + '/moving-mean'] = movingmean
                    # res_bias[clean_name(layer.name) + '/moving-var'] = movingvariance
                    pass
                elif len(weights) == 1:
                    kernel = weights[0]
                    res_kernel[clean_name(layer.name) + '/kernel'] = kernel
                else:
                    raise ValueError('how many '.format(len(weights)))
        return res_kernel, res_bias

    def get_act(self):
        # also used when statistic need to be calc
        val_data = [self.dataset.x_test_ref, self.dataset.y_test_ref, np.ones_like(self.validation_data[2]),
                    np.zeros_like(self.validation_data[3])]
        # val_data = self.validation_data

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

            # t=self.sess.run(self.act_l, feed_dict=feed_dict)
            # len(t.keys())
            # np.unique(t.keys())[:4]
            #
            # t['Layer12/dense/in'].shape
            # t['Layer12/dense/act'].shape

            for _act_name, _act in self.sess.run(self.act_l, feed_dict=feed_dict).iteritems():
                if _act_name in res:
                    res[_act_name] = np.concatenate((res[_act_name], _act), axis=0)
                else:
                    res[_act_name] = _act
            i += self.batch_size
        return res

    def get_act_param_summ_str(self):
        val_data = [self.dataset.x_test_ref, self.dataset.y_test_ref, np.ones_like(self.validation_data[2]),
                    np.zeros_like(self.validation_data[3])]
        # val_data = self.validation_data
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
                if 'act' in name and 'ptrate' in name and 'softmax' in name:
                    logger.debug('{} {} {}  is write into df'.format(name, _iter, val))
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


def schedule(epoch, x=(30., 100.), y=(10., 10.), init=0.01):
    if x is None:
        return init
    if not isinstance(x, tuple) and not isinstance(x, list):
        x = [x]
    if not isinstance(y, tuple) and not isinstance(y, list):
        y = [y]
    x = [float(_x) for _x in x]
    y = [float(_y) for _y in y]
    func_l = [float(init)]
    for _y in y:
        func_l.append(func_l[-1] / _y)
    x = [0] + x + [99999]
    for ind, _x in enumerate(x):
        if epoch >= x[ind] and epoch < x[ind + 1]: break

    return func_l[ind]


if __name__ == '__main__':
    y = [schedule(i, x=(150,), y=(10,), init=0.01) for i in range(0, 999, 2)]
    from vis_utils import *

    plt.plot(y)
    plt.show()
