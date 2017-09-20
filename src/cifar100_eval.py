import tensorflow as tf
import tensorflow.contrib
from datasets import cifar100
from model import resnet101, resnet50
from datasets.cifar100 import load_batch, c2f_map

import tensorflow.contrib.slim as slim
from dynamics import svd
import utils, numpy as np

utils.init_dev(utils.get_dev())
from hypers import cifar100_eval as FLAGS

f2c_map = {}
for c, fs in c2f_map.items():
    for f in fs:
        f2c_map[f] = c
f2c_arr = np.array(f2c_map.values())
c2f_arr = np.array([list(v) for v in c2f_map.values()])


def c2f(c):
    return c2f_arr[c, :]


def f2c(f):
    return f2c_arr[f]


def main():
    utils.mkdir_p(FLAGS.log_dir, delete=True)
    # load the dataset
    dataset = cifar100.get_split('test', FLAGS.data_dir, )

    # load batch
    batch_queue = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=False)
    images, labels = batch_queue.dequeue()
    # get the model prediction
    predictions, end_points = resnet50(images, classes=100)
    global logits, fc_in, fc_weight
    fc_weight = tf.squeeze(slim.get_variables('.*/logits/weights')[0], name='dynamic/weight')
    tf.summary.histogram('weight/single_val', svd.Svd(fc_weight)[0])
    tf.summary.scalar('weight/sparsity/zero_frac', tf.nn.zero_fraction(fc_weight))
    tf.summary.histogram('weight/sparsity/hist', fc_weight)
    # todo ortho use t-sne

    fc_in = tf.squeeze(end_points['resnet_v2_50/block4'], name='dynamic/fc_in')
    logits = tf.squeeze(end_points['resnet_v2_50/logits'], name='dynamic/logits')

    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)

    loss_100 = tf.losses.softmax_cross_entropy(
        logits=predictions,
        onehot_labels=one_hot_labels)

    labels_coarse = tf.py_func(f2c, [labels], tf.int64)
    # labels_coarse= tf.reshape(tf.concat(tf.constant(1,tf.int64), labels_coarse), [ 1,2])
    labels_coarse = tf.reshape(labels_coarse, labels.shape)
    labels_fine = tf.py_func(c2f, [labels_coarse], tf.int64)
    labels_fine = tf.reshape(labels_fine, labels.shape.as_list() + [5, ])

    one_hot_labels_coarse = tf.reduce_sum(
        tf.reshape(
            tf.to_int64(
                slim.one_hot_encoding(
                    tf.reshape(labels_fine, (-1,)),
                    num_classes=dataset.num_classes
                )
            ),
            labels.shape.as_list() + [5, -1]),
        axis=1
    )

    loss_20 = tf.losses.log_loss(
        predictions=tf.nn.softmax(predictions),
        labels=one_hot_labels_coarse,
        weights=FLAGS.beta,
        loss_collection=None if not FLAGS.multi_loss else tf.GraphKeys.LOSSES
    )

    bs = labels_fine.shape[0]
    predictions_l = []
    for ind in range(5):
        sel = tf.stack([tf.range(bs, dtype=tf.int64), labels_fine[:, ind]], axis=1)
        predictions_l.append(tf.gather_nd(predictions, sel))
    predictions_group = tf.stack(predictions_l, axis=1)

    labels_group_one_hot = tf.equal(labels_fine, tf.expand_dims(labels, axis=-1))
    labels_group_one_hot = tf.to_int64(labels_group_one_hot)

    loss_group = tf.losses.softmax_cross_entropy(
        logits=predictions_group,
        onehot_labels=labels_group_one_hot,
        weights=FLAGS.gamma,
        loss_collection=None if not FLAGS.multi_loss else tf.GraphKeys.LOSSES
    )

    print '>> loss', tf.losses.get_losses(), len(tf.losses.get_losses())

    loss_reg = tf.add_n(tf.losses.get_regularization_losses())

    total_loss = tf.losses.get_total_loss()

    tf.summary.scalar('loss/20/val/ori', loss_20)

    # streaming metrics to evaluate
    predictions = tf.to_int64(tf.argmax(predictions, 1))
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map(
        {
            # 'mse/val': slim.metrics.streaming_mean_squared_error(predictions, labels),
            'acc/val': slim.metrics.streaming_accuracy(predictions, labels),
            'loss/ttl/val': slim.metrics.streaming_mean(total_loss),
            'loss/100/val': slim.metrics.streaming_mean(loss_100),
            'loss/20/val': slim.metrics.streaming_mean(loss_20),
            'loss/reg/val': slim.metrics.streaming_mean(loss_reg),
            'loss/group/ttl/val': slim.metrics.streaming_mean(loss_group)
        })

    # write the metrics as summaries
    for metric_name, metric_value in metrics_to_values.iteritems():
        tf.summary.scalar(metric_name + '/values', metric_value)

    global writer
    writer = tf.summary.FileWriter(FLAGS.log_dir + '/custom')

    class ExampleHook(tf.train.SessionRunHook):

        # self.fc_in, self.fc_weight, self.logits = fc_in, fc_weight, logits
        # self.fc_in_t,self.fc_weight_t,self.logits_t=[],[],[]

        def begin(self):
            self.fc_in_t, self.fc_weight_t, self.logits_t = [], None, []
            self.step = None
            # print('Starting the session.')

            # def after_create_session(self, session, coord):
            # When this is called, the graph is finalized and
            # ops can no longer be added to the graph.
            # print('Session created.')

        def before_run(self, run_context):
            # print('Before calling session.run().')
            return tf.train.SessionRunArgs(
                fetches=[fc_in, fc_weight, logits, slim.get_or_create_global_step()])

        def after_run(self, run_context, run_values):
            # print('Done running one step. The value of my tensor')
            if self.fc_weight_t is None:
                self.fc_weight_t = run_values.results[1]
            if len(self.fc_in_t) <= 3500 // FLAGS.batch_size:
                self.fc_in_t.append(run_values.results[0])
                self.logits_t.append(run_values.results[2])

            self.step = run_values.results[3]

        def end(self, session):
            from dynamics.stats import ActStat, KernelStat

            ker_stat = KernelStat()
            write_single_val(ker_stat.ortho(self.fc_weight_t), self.step, 'weight/ortho', writer)
            write_single_val(ker_stat.orthoabs(self.fc_weight_t), self.step, 'weight/ortho/abs', writer)
            write_single_val(ker_stat.zerofrac(self.fc_weight_t), self.step, 'weight/zero_frac/cpu', writer)

            act_stat = ActStat()
            write_single_val(act_stat.orthochnl(np.stack(self.fc_in_t, axis=0)), self.step, 'fcin/ortho', writer)
            write_single_val(act_stat.orthochnlabs(np.stack(self.fc_in_t, axis=0)), self.step, 'fcin/ortho/abs', writer)
            write_single_val(act_stat.orthochnl(np.stack(self.logits_t, axis=0)), self.step, 'logits/ortho', writer)
            write_single_val(act_stat.orthochnlabs(np.stack(self.logits_t, axis=0)), self.step, 'logits/ortho/abs',
                             writer)

            # print('Done with the session.')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    slim.evaluation.evaluation_loop(
        '',
        FLAGS.checkpoint_dir,
        FLAGS.log_dir,
        num_evals=FLAGS.num_evals,
        eval_op=metrics_to_updates.values(),
        session_config=_sess_config,
        eval_interval_secs=FLAGS.eval_interval_secs,
        hooks=[ExampleHook(),
               tf.contrib.training.SummaryAtEndHook(
                   log_dir=FLAGS.log_dir,
                   summary_writer=writer,
                   summary_op=None,
                   feed_dict=None)]
    )

    #


def write_single_val(value, epoch_iter, name, writer):
    summary = tf.Summary()
    summary_value = summary.value.add()

    summary_value.simple_value = value
    summary_value.tag = name
    writer.add_summary(summary, epoch_iter)


class HistWriter():
    def __init__(self):
        _sess_config = tf.ConfigProto(allow_soft_placement=True)
        _sess_config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=_sess_config)

        # Initializes function that converts PNG to JPEG data.
        self.tensor = tf.placeholder(dtype=tf.string)

    def write(self, ndarray, epoch_iter, name, writer):
        self.summ = tf.summary.histogram(name, self.tensor)
        summ_str = self._sess.run(self.summ, feed_dict={self.tensor: ndarray})
        writer.add_summary(summ_str, epoch_iter)


if __name__ == '__main__':
    main()
