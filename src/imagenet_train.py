import tensorflow as tf
from deployment import model_deploy
from datasets import imagenet
from datasets.imagenet import load_batch

# from model import resnet101_2 as resnet101
from model import resnet101
import utils, numpy as np
import tensorflow.contrib.slim as slim

from hypers import imagenet as FLAGS


def clone_fn(batch_queue):
    images, labels = batch_queue.dequeue()
    tf.summary.image('input/img', images)
    predictions, end_points = resnet101(images, classes=FLAGS.nclasses)

    one_hot_labels = slim.one_hot_encoding(
        labels,
        FLAGS.nclasses)
    tf.logging.info('>> dataset has class:{}'.format(FLAGS.nclasses))

    loss_100 = tf.losses.softmax_cross_entropy(
        logits=predictions,
        onehot_labels=one_hot_labels)

    loss_reg = tf.reduce_sum(tf.losses.get_regularization_losses()) + tf.constant(1e-7,tf.float32)

    tf.logging.info('loss are {}'.format(tf.losses.get_losses()))

    total_loss = tf.losses.get_total_loss()

    # todo many be add summaries afterwards
    tf.summary.scalar('loss/ttl/train', total_loss)
    tf.summary.scalar('loss/100/train', loss_100)
    tf.summary.scalar('loss/reg/train', loss_reg)

    acc = slim.metrics.accuracy(
        predictions=tf.to_int64(tf.argmax(predictions, 1)),
        labels=labels
    )

    return end_points, [total_loss, loss_100, loss_reg, acc]


def get_init_fn():
    slim.assign_from_checkpoint_fn(
        FLAGS.checkpoint_path,
        slim.get_variables_to_restore(
            exclude=[".*logits.*", ".*Ftrl.*", '.*Momentum.*', '.*fully_connected.*', '.*global_step.*']),
        ignore_missing_vars=False
    )


def train_step_fn(session, *args, **kwargs):
    from tensorflow.contrib.slim.python.slim.learning import train_step

    total_loss, should_stop = train_step(session, *args, **kwargs)

    if train_step_fn.step % 20 == 0:
        acc_ = session.run(train_step_fn.acc)

        print('>> Step %s - Loss: %.2f  acc: %.2f%%' % (
            str(train_step_fn.step).rjust(6, '0'), total_loss,
            np.mean(acc_) * 100))

    train_step_fn.step += 1
    return [total_loss, should_stop]


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.get_default_graph().as_default():
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=False,
            replica_id=0,
            num_replicas=1,
            num_ps_tasks=0,
        )
        with tf.device(deploy_config.variables_device()):
            global_step = slim.get_or_create_global_step()

        dataset = imagenet.get_split('train', FLAGS.data_dir)

        with tf.device(deploy_config.inputs_device()):
            # load batch of dataset
            batch_queue = load_batch(
                dataset,
                FLAGS.batch_size,
                is_training=True)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        end_points, [total_loss, loss_100, loss_reg, acc] = clones[0].outputs
        # todo can add summarys here
        summaries = summaries.union(
            {tf.summary.scalar('loss/ttl/train', total_loss),
             tf.summary.scalar('loss/100/train', loss_100),
             tf.summary.scalar('loss/reg/train', loss_reg),
             tf.summary.scalar('acc/train', acc)
             })

        with tf.device(deploy_config.optimizer_device()):
            global_step = slim.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(
                FLAGS.init_lr, global_step,
                FLAGS.lr_decay_per_steps, FLAGS.lr_decay, staircase=True)
            summaries.add(
                slim.summary.scalar('lr', learning_rate)
            )
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

        total_loss, clone_gradients = model_deploy.optimize_clones(
            clones, optimizer,
        )

        grad_updates = optimizer.apply_gradients(clone_gradients, global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        train_step_fn.step = 0
        train_step_fn.acc = acc

        _sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        _sess_config.gpu_options.allow_growth = True
        slim.learning.train(
            train_tensor,
            session_config=_sess_config,
            logdir=FLAGS.log_dir,
            init_fn=get_init_fn(),
            train_step_fn=train_step_fn,
            summary_op=summary_op,
            number_of_steps=FLAGS.nsteps,
            save_interval_secs=FLAGS.interval,
            save_summaries_secs=FLAGS.interval/10,
            log_every_n_steps=20,
            # trace_every_n_steps=20,
        )


if __name__ == '__main__':
    utils.rm(FLAGS.log_dir)
    utils.init_dev(utils.get_dev(n=FLAGS.num_clones))
    import multiprocessing as mp, imagenet_eval, time

    mp.Process(target=main).start()
    time.sleep(FLAGS.interval//2)

    proc = mp.Process(target=imagenet_eval.main, args=())
    proc.start()
