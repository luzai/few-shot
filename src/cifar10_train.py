import tensorflow as tf

from datasets import cifar10
from model import resnet101
from datasets.cifar10 import load_batch
import utils, numpy as np

from hypers import cifar10 as FLAGS

import tensorflow.contrib.slim as slim

utils.init_dev(utils.get_dev())


def main(args):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # load the dataset
    dataset = cifar10.get_split('train', FLAGS.data_dir)

    # load batch of dataset
    batch_queue = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=True)
    images, labels = batch_queue.dequeue()
    # run the image through the model

    predictions, _ = resnet101(images)

    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)
    tf.losses.softmax_cross_entropy(
        logits=predictions,
        onehot_labels=one_hot_labels)
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('loss/train', total_loss)

    learning_rate = tf.train.exponential_decay(
        FLAGS.init_lr, slim.get_or_create_global_step(),
        FLAGS.lr_decay_per_steps, FLAGS.lr_decay,
        staircase=True)
    slim.summary.scalar('lr', learning_rate)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    # create train op
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        summarize_gradients=False,
        # variables_to_train=slim.get_variables('resnet_v2_101/logits'),
    )

    variables_to_restore = slim.get_variables_to_restore(
        exclude=[".*logits.*", '.*global_step.*'])

    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
        FLAGS.checkpoint_path, variables_to_restore)

    global_step = slim.get_or_create_global_step()
    global_step_init = tf.assign(global_step, 0)

    acc = slim.metrics.accuracy(tf.to_int64(tf.argmax(predictions, 1)), labels)
    slim.summary.scalar('acc/train', acc)

    def InitAssignFn(sess):
        print 'init from pretrained model'
        sess.run([init_assign_op, global_step_init], init_feed_dict)

    def train_step_fn(session, *args, **kwargs):
        from tensorflow.contrib.slim.python.slim.learning import train_step

        total_loss, should_stop = train_step(session, *args, **kwargs)

        if train_step_fn.step % 20 == 0:
            acc_ = session.run([train_step_fn.acc])
            # print acc_
            print('Step {} - Loss: {} acc:{}%'.format(
                str(train_step_fn.step).rjust(6, '0'), total_loss,
                np.mean(acc_) * 100))

        train_step_fn.step += 1
        return [total_loss, should_stop]

    train_step_fn.step = 0
    train_step_fn.acc = acc

    # run training

    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    slim.learning.train(
        train_op,
        FLAGS.log_dir,
        init_fn=InitAssignFn,
        save_summaries_secs=60,
        save_interval_secs=60,
        session_config=_sess_config,
        number_of_steps=None,
        log_every_n_steps=20,
        train_step_fn=train_step_fn,
        trace_every_n_steps=None,
    )


if __name__ == '__main__':
    utils.rm(FLAGS.log_dir)
    import multiprocessing as mp
    import cifar10_eval

    proc = mp.Process(target=cifar10_eval.main, args=())
    proc.start()
    # proc = utils.shell('python cifar10_eval.py', block=False)
    tf.app.run()
