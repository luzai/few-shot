import tensorflow as tf

from datasets import cifar10
from model import resnet101
from datasets.cifar10 import load_batch

import tensorflow.contrib.slim as slim

import utils

utils.init_dev(utils.get_dev())
from hypers import cifar10_eval as FLAGS


def main():
    utils.rm(FLAGS.log_dir)

    # load the dataset
    dataset = cifar10.get_split('test', FLAGS.data_dir, )

    # load batch
    batch_queue = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=False)
    images, labels = batch_queue.dequeue()
    # get the model prediction
    predictions, _ = resnet101(images)

    # convert prediction values for each class into single class prediction

    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)
    tf.losses.softmax_cross_entropy(
        logits=predictions,
        onehot_labels=one_hot_labels)

    total_loss = tf.losses.get_total_loss()
    # tf.summary.scalar('loss/val/ori', total_loss)

    # streaming metrics to evaluate
    predictions = tf.to_int64(tf.argmax(predictions, 1))
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
        # 'mse/val': slim.metrics.streaming_mean_squared_error(predictions, labels),
        'acc/val': slim.metrics.streaming_accuracy(predictions, labels),
        'loss/val': slim.metrics.streaming_mean(total_loss)
    })

    # write the metrics as summaries
    for metric_name, metric_value in metrics_to_values.iteritems():
        tf.summary.scalar(metric_name + '/values', metric_value)
    # for metric_name, metric_value in metrics_to_updates.iteritems():
    #     tf.summary.scalar(metric_name+'/update', metric_value)

    # evaluate on the model saved at the checkpoint directory
    # evaluate every eval_interval_secs
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
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
    main()
