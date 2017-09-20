import tensorflow as tf

from datasets import imagenet
from model import resnet101, resnet50
from datasets.imagenet import load_batch

import tensorflow.contrib.slim as slim
from dynamics import svd
import utils, numpy as np
from hypers import imgnet_eval as FLAGS


def main():
    utils.init_dev(utils.get_dev())
    utils.mkdir_p(FLAGS.log_dir, delete=True)
    # load the dataset
    dataset = imagenet.get_split('validation', FLAGS.data_dir, )

    # load batch
    batch_queue = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=False)
    images, labels = batch_queue.dequeue()
    # get the model prediction
    predictions, end_points = resnet101(images, classes=FLAGS.nclasses)
    global logits, fc_in, fc_weight
    fc_weight = tf.squeeze(slim.get_variables('.*/logits/weights')[0], name='dynamic/weight')
    tf.summary.histogram('weight/single_val', svd.Svd(fc_weight)[0])
    tf.summary.scalar('weight/sparsity/zero_frac', tf.nn.zero_fraction(fc_weight))
    tf.summary.histogram('weight/sparsity/hist', fc_weight)
    # todo ortho use t-sne

    fc_in = tf.squeeze(end_points['resnet_v2_101/block4'], name='dynamic/fc_in')
    logits = tf.squeeze(end_points['resnet_v2_101/logits'], name='dynamic/logits')

    # convert prediction values for each class into single class prediction
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)
    assert dataset.num_classes == FLAGS.nclasses, 'classes equal'
    tf.losses.softmax_cross_entropy(
        logits=predictions,
        onehot_labels=one_hot_labels)

    loss_100 = tf.losses.softmax_cross_entropy(
        logits=predictions,
        onehot_labels=one_hot_labels)

    loss_reg = tf.add_n(tf.losses.get_regularization_losses())+tf.constant(1e-7,tf.float32)

    total_loss = tf.losses.get_total_loss()

    # streaming metrics to evaluate
    predictions = tf.to_int64(tf.argmax(predictions, 1))
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map(
        {
            'acc/val': slim.metrics.streaming_accuracy(predictions, labels),
            'loss/ttl/val': slim.metrics.streaming_mean(total_loss),
            'loss/100/val': slim.metrics.streaming_mean(loss_100),
            'loss/reg/val': slim.metrics.streaming_mean(loss_reg), }
    )

    # write the metrics as summaries
    for metric_name, metric_value in metrics_to_values.iteritems():
        tf.summary.scalar(metric_name + '/values', metric_value)

    global writer
    writer = tf.summary.FileWriter(FLAGS.log_dir + '/custom')

    class ExampleHook(tf.train.SessionRunHook):
        def __init__(self):
            self.summary_times = 0

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

            if self.step is None:
                self.step = run_values.results[3]

            if len(self.fc_in_t) <= 3500 // FLAGS.batch_size:
                self.fc_in_t.append(run_values.results[0])
                self.logits_t.append(run_values.results[2])
            elif not self.summary_times % 100: # I gusee evaluation size is not sensitive # todo verify labels are consisten across multi evaluation
                run_context.request_stop()
            self.summary_times += 1

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
               # tf.contrib.training.SummaryAtEndHook(
               #     log_dir=FLAGS.log_dir,
               #     summary_writer=writer,
               #     summary_op=None,
               #     feed_dict=None)
               ]
    )


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
