# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts cifar10 data to TFRecords of TF-Example protos.

This module downloads the cifar10 data, uncompresses it, reads the files
that make up the cifar10 data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take several minutes to run.

"""

from __future__ import division
from __future__ import absolute_import

import os
import sys
import tarfile
import utils
import numpy as np
from six.moves import cPickle, urllib
import tensorflow as tf

from datasets import dataset_utils
from datasets.cifar100 import coarse_labels_human, fine_labels_human
from datasets import cifar100
from utils import root_path

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_dir', root_path + '/data/cifar100', 'cifar100 data dir')

# The URL where the CIFAR data can be downloaded.
_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

# The height and width of each image.
_IMAGE_SIZE = 32


def _add_to_tfrecord(filename, tfrecord_writer, offset=0):
    """Loads data from the cifar10 pickle files and writes files to a TFRecord.

    Args:
      filename: The filename of the cifar10 pickle file.
      tfrecord_writer: The TFRecord writer to use for writing.
      offset: An offset into the absolute number of images previously written.

    Returns:
      The new offset.
    """
    with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info < (3,):
            data = cPickle.load(f)
        else:
            data = cPickle.load(f, encoding='bytes')

    images = data[b'data']
    num_images = images.shape[0]

    images = images.reshape((num_images, 3, 32, 32))
    labels = data[b'fine_labels']
    coarse_labels = data[b'coarse_labels']

    c2f_map = {}
    for lb, cl in zip(labels, coarse_labels):
        if cl not in c2f_map:
            c2f_map[cl] = {lb}
        else:
            c2f_map[cl].add(lb)

    utils.pickle(c2f_map, utils.root_path + '/data/cifar100/c2f_map.pkl')
    b2a_map = {}
    ind = 0
    for c, fs in c2f_map.items():
        for f in fs:
            b2a_map[f] = ind
            ind += 1
    utils.pickle(b2a_map, utils.root_path + '/data/cifar100/b2a_map.pkl')
    a2b_map = {a: b for b, a in b2a_map.items()}

    # labels = [b2a_map[lb] for lb in labels]

    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_png(image_placeholder)

        with tf.Session() as sess:
            for j in range(num_images):
                sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
                    filename, offset + j + 1, offset + num_images))
                sys.stdout.flush()

                image = np.squeeze(images[j]).transpose((1, 2, 0))
                lb = labels[j]

                png_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: image})

                example = dataset_utils.image_to_tfexample(
                    png_string, b'png', _IMAGE_SIZE, _IMAGE_SIZE, lb)
                tfrecord_writer.write(example.SerializeToString())

    return offset + num_images


def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      split_name: The name of the train/test split.

    Returns:
      An absolute file path.
    """
    return '%s/cifar100_%s.tfrecord' % (dataset_dir, split_name)


def _clean_up_temporary_files(dataset_dir):
    """Removes temporary files used to create the dataset.

    Args:
      dataset_dir: The directory where the temporary files are stored.
    """
    filename = _DATA_URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)

    tmp_dir = os.path.join(dataset_dir, 'cifar-10-batches-py')
    tf.gfile.DeleteRecursively(tmp_dir)


def run(args):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    dataset_dir = FLAGS.dataset_dir

    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    training_filename = _get_output_filename(dataset_dir, 'train')
    testing_filename = _get_output_filename(dataset_dir, 'test')

    # if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
    #     print('Dataset files already exist. Exiting without re-creating them.')
    #     return

    dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)

    # First, process the training data:
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        offset = 0

        filename = os.path.join(dataset_dir,
                                'cifar-100-python', 'train')  # 1-indexed.
        offset = _add_to_tfrecord(filename, tfrecord_writer, offset)

    # Next, process the testing data:
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        filename = os.path.join(dataset_dir,
                                'cifar-100-python',
                                'test')
        _add_to_tfrecord(filename, tfrecord_writer)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(fine_labels_human)), fine_labels_human))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    labels_to_class_names = dict(zip(range(len(coarse_labels_human)), coarse_labels_human))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir, filename='labels-coarse.txt')

    # _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the Cifar100 dataset!')


if __name__ == '__main__':
    exit(-1)# do not run! when training
    tf.app.run(main=run)
