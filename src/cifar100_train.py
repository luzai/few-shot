import tensorflow as tf
import os

from datasets import cifar100
from datasets.cifar100 import load_batch, c2f_map
from model import resnet101, resnet50
import utils, numpy as np
import tensorflow.contrib.slim as slim

utils.init_dev(utils.get_dev())
from hypers import cifar100 as FLAGS

# def map_label(input_tensor):
#     keys = np.array(cifar100.mapp.keys(), dtype=np.int64)
#     values = np.array(cifar100.mapp.values(), dtype=np.int64)
#     table = tf.contrib.lookup.HashTable(
#         tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
#     out = table.lookup(input_tensor)
#     return out

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
    # train_op, InitAssignFn, train_step_fn = config_graph()

    tf.logging.set_verbosity(tf.logging.DEBUG)
    # load the dataset
    dataset = cifar100.get_split('train', FLAGS.data_dir)

    # load batch of dataset
    batch_queue = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=True)
    images, labels = batch_queue.dequeue()
    slim.summary.image('input/image', images)

    # run the image through the model
    predictions, end_points = resnet50(images, classes=dataset.num_classes)

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
    # ema = tf.train.ExponentialMovingAverage(decay=0.9)
    # total_loss_avg_op = ema.apply([total_loss])
    # total_loss_avg = ema.average(total_loss)
    slim.summary.scalar('loss/ttl/train', total_loss)
    # slim.summary.scalar('loss/total_avg', total_loss_avg)
    slim.summary.scalar('loss/100/train', loss_100)
    slim.summary.scalar('loss/reg/train', loss_reg)
    slim.summary.scalar('loss/20/train', loss_20)
    slim.summary.scalar('loss/group/total/train', loss_group)

    global_step = slim.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        FLAGS.init_lr, global_step,
        FLAGS.lr_decay_per_steps, FLAGS.lr_decay, staircase=True)
    slim.summary.scalar('lr', learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    # create train op
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
    )
    # with tf.control_dependencies([total_loss]):
    #     train_op = tf.group(train_op, total_loss_avg_op)

    variables_to_restore = slim.get_variables_to_restore(
        exclude=[".*logits.*", ".*Ftrl.*", '.*Momentum.*', 'global_step'])

    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
        FLAGS.checkpoint_path, variables_to_restore)

    def InitAssignFn(sess):
        print 'init from pretrained model'
        sess.run([init_assign_op, ], init_feed_dict)

    acc = slim.metrics.accuracy(tf.to_int64(tf.argmax(predictions, 1)), labels)

    slim.summary.scalar('acc/train', acc)

    def train_step_fn(session, *args, **kwargs):
        from tensorflow.contrib.slim.python.slim.learning import train_step

        total_loss_, should_stop = train_step(session, *args, **kwargs)
        # total_loss_ = 0
        if train_step_fn.step % 196 == 0:
            acc_ = session.run(train_step_fn.acc)
            print acc_
            print('>> Step {} - Loss: {} acc: {}%'.format(
                str(train_step_fn.step).rjust(6, '0'), total_loss_,
                np.mean(acc_) * 100))

        train_step_fn.step += 1
        return [total_loss_, should_stop]

    train_step_fn.step = 0
    train_step_fn.acc = acc

    # run training

    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    slim.learning.train(
        train_op,
        FLAGS.log_dir,
        init_fn=InitAssignFn,
        save_summaries_secs=FLAGS.interval / 10,
        save_interval_secs=FLAGS.interval,
        session_config=_sess_config,
        number_of_steps=None,
        log_every_n_steps=196,
        train_step_fn=train_step_fn,
        # trace_every_n_steps=50,
    )


if __name__ == '__main__':
    utils.rm(FLAGS.log_dir)
    import multiprocessing as mp
    import cifar100_eval, time

    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/extras/CUPTI/lib64:' + os.environ['LD_LIBRARY_PATH']
    proc = mp.Process(target=cifar100_eval.main, args=())
    proc.start()
    # time.sleep(FLAGS.interval//2)
    # mp.Process(target=main).start()
    main()
