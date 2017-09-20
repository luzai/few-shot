# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Cifar10 dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_cifar10.py
"""
from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf
import numpy as np
from utils import root_path
from utils import *

# try:
from datasets import dataset_utils
from preprocessing import cifar_preprocessing

import tensorflow.contrib.slim as slim

_FILE_PATTERN = 'cifar100_%s.tfrecord'

SPLITS_TO_SIZES = {'train': 50000, 'test': 10000}

_NUM_CLASSES = 100

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 32 x 3] color image.',
    'label': 'A single integer between 0 and 99',
}

coarse_labels_human = np.array(['aquatic_mammals',
                                'fish',
                                'flowers',
                                'food_containers',
                                'fruit_and_vegetables',
                                'household_electrical_devices',
                                'household_furniture',
                                'insects',
                                'large_carnivores',
                                'large_man-made_outdoor_things',
                                'large_natural_outdoor_scenes',
                                'large_omnivores_and_herbivores',
                                'medium_mammals',
                                'non-insect_invertebrates',
                                'people',
                                'reptiles',
                                'small_mammals',
                                'trees',
                                'vehicles_1',
                                'vehicles_2'])

fine_labels_human = np.array([['apple', 'aquarium_fish', 'baby', 'bear', 'beaver'],
                              ['bed', 'bee', 'beetle', 'bicycle', 'bottle'],
                              ['bowl', 'boy', 'bridge', 'bus', 'butterfly'],
                              ['camel', 'can', 'castle', 'caterpillar', 'cattle'],
                              ['chair', 'chimpanzee', 'clock', 'cloud', 'cockroach'],
                              ['couch', 'crab', 'crocodile', 'cup', 'dinosaur'],
                              ['dolphin', 'elephant', 'flatfish', 'forest', 'fox'],
                              ['girl', 'hamster', 'house', 'kangaroo', 'keyboard'],
                              ['lamp', 'lawn_mower', 'leopard', 'lion', 'lizard'],
                              ['lobster', 'man', 'maple_tree', 'motorcycle', 'mountain'],
                              ['mouse', 'mushroom', 'oak_tree', 'orange', 'orchid'],
                              ['otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree'],
                              ['plain', 'plate', 'poppy', 'porcupine', 'possum'],
                              ['rabbit', 'raccoon', 'ray', 'road', 'rocket'],
                              ['rose', 'sea', 'seal', 'shark', 'shrew'],
                              ['skunk', 'skyscraper', 'snail', 'snake', 'spider'],
                              ['squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table'],
                              ['tank', 'telephone', 'television', 'tiger', 'tractor'],
                              ['train', 'trout', 'tulip', 'turtle', 'wardrobe'],
                              ['whale', 'willow_tree', 'wolf', 'woman', 'worm']]).flatten()

b2a_map = unpickle(root_path+'/data/cifar100/b2a_map.pkl')
c2f_map = unpickle(root_path+'/data/cifar100/c2f_map.pkl')
a2b_map = {a:b for b,a in b2a_map.items()}

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading cifar10.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=[32, 32, 3]),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names)


def load_batch(dataset, batch_size, height=32, width=32, is_training=False):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=16,
        common_queue_capacity=40 * batch_size,
        common_queue_min=30 * batch_size)

    image, label = data_provider.get(['image', 'label'])

    image = cifar_preprocessing.preprocess_image(
        image,
        height,
        width,
        is_training)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        # allow_smaller_final_batch=True,
        num_threads=256,
        capacity=30 * batch_size,
    )
    batch_queue = slim.prefetch_queue.prefetch_queue(
        [images, labels], num_threads=256,
        capacity=10 * batch_size,
        name='prefetch/train' if is_training else 'prefetch/val'
    )
    return batch_queue
