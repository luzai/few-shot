from loader import *
from datasets import *
from vis_utils import *
from plt_utils import *
from utils import *
from stats import *
from logs import logger
import logs, datasets, vis_utils, loader

logger.setLevel(logs.WARN)
matplotlib.style.use('ggplot')

# utils.init_dev(utils.get_dev(ok=()))
# utils.allow_growth()
import uuid

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def get_iter(x_train, y_train):
  batch_size = 128
  ind = 0
  while True:
    yield x_train[ind:ind + batch_size], y_train[ind:ind + batch_size]
    ind += batch_size
    if (ind + batch_size) >= x_train.shape[0]:
      ind %= x_train.shape[0]


def svd(tensor):
  _, s, _ = np.linalg.svd(tensor, full_matrices=True)
  return s.sum()


data = get_iter(x_train, y_train)

tf.reset_default_graph()
x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
nn = tf.layers.dense(x, 512, activation=tf.nn.relu)
nn = tf.layers.dense(nn, 512, activation=tf.nn.relu)
nn = tf.layers.dense(nn, 10)

y = tf.placeholder(shape=[None, 10], dtype=tf.float32)
cost = tf.losses.softmax_cross_entropy(
    onehot_labels=y, logits=nn) + tf.py_func(svd, [nn], tf.float32)

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss=cost)

init = tf.global_variables_initializer()

tf.summary.scalar("cost", cost)
merged_summary_op = tf.summary.merge_all()

with utils.get_session() as sess:
  sess.run(init)
  uniq_id = "../tfevents/" + uuid.uuid1().__str__()[:6]
  print os.getcwd(), uniq_id
  summary_writer = tf.summary.FileWriter(uniq_id, graph=tf.get_default_graph())
  
  for step in range(10000):
    x_, y_ = data.next()
    # print x_.shape,y_.shape
    _, val, summary = sess.run([optimizer, cost, merged_summary_op],
                               feed_dict={x: x_, y: y_})
    if step % 50 == 0:
      summary_writer.add_summary(summary, step)
    if step % 500 == 0:
      print step, val
