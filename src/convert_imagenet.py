from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import numpy as np
import tensorflow as tf
import utils
from utils import *

tf.app.flags.DEFINE_string('train_directory', utils.root_path+'/data/imagenet-raw',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', utils.root_path+'/data/imagenet-raw',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', utils.root_path+'/data/imagenet600',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 1024,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 128,  # 128
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 128,# 64
                            'Number of threads to preprocess the images.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   n01440764
#   n01443537
#   n01484850
# where each line corresponds to a label expressed as a synset. We map
# each synset contained in the file to an integer (based on the alphabetical
# ordering). See below for details.
tf.app.flags.DEFINE_string('labels_file',
                           utils.root_path+'/data/imagenet22k.txt',
                           'Labels file')

# This file containing mapping from synset to human-readable label.
# Assumes each line of the file looks like:
#
#   n02119247    black fox
#   n02119359    silver fox
#   n02119477    red fox, Vulpes fulva
#
# where each line corresponds to a unique mapping. Note that each line is
# formatted as <synset>\t<human readable label>.
tf.app.flags.DEFINE_string('imagenet_metadata_file',
                           utils.root_path+'/data/words.txt',
                           'ImageNet metadata file')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, synset, human,
                        height, width):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
      human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
      bbox: list of bounding boxes; each box is a list of integers
        specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
        the same label as the image label.
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/synset': _bytes_feature(synset),
        'image/class/text': _bytes_feature(human),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        _sess_config = tf.ConfigProto(allow_soft_placement=True)
        _sess_config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=_sess_config)

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                              feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})

        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image



def _is_png(filename):
    """Determine if a file contains a PNG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a PNG.
    """
    # File list from:
    # https://groups.google.com/forum/embed/?place=forum/torch7#!topic/torch7/fOSTXHIESSU
    return 'n02105855_2933.JPEG' in filename


def _is_cmyk(filename):
    """Determine if file contains a CMYK JPEG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a JPEG encoded with CMYK color space.
    """
    # File list from:
    # https://github.com/cytsai/ilsvrc-cmyk-image-list
    blacklist = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                 'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                 'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                 'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                 'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                 'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                 'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                 'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                 'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                 'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                 'n07583066_647.JPEG', 'n13037406_4650.JPEG']
    return filename.split('/')[-1] in blacklist


def _process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    try:
        image_data = tf.gfile.FastGFile(filename, 'r').read()

        # Clean the dirty data.
        if _is_png(filename):
            # 1 image is a PNG.
            print('Converting PNG to JPEG for %s' % filename)
            image_data = coder.png_to_jpeg(image_data)
        elif _is_cmyk(filename):
            # 22 JPEG images are in CMYK colorspace.
            print('Converting CMYK to RGB for %s' % filename)
            image_data = coder.cmyk_to_rgb(image_data)

        # Decode the RGB JPEG.

        image = coder.decode_jpeg(image_data)
    except Exception as inst:
        utils.rm(filename)
        tf.logging.error(inst)
        tf.logging.error(filename)
        append_file(filename)
        # raise ValueError('rm file')

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               synsets, labels, humans, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      synsets: list of strings; each string is a unique WordNet ID
      labels: list of integer; each integer identifies the ground truth
      humans: list of strings; each string is a human-readable label
      bboxes: list of bounding boxes for each image. Note that each entry in this
        list might contain from 0+ entries corresponding to the number of bounding
        box annotations for the image.
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            synset = synsets[i]
            human = humans[i]

            image_buffer, height, width = _process_image(filename, coder)

            example = _convert_to_example(filename, image_buffer, label,
                                          synset, human,
                                          height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, synsets, labels, humans,
                         num_shards):
    """Process and save list of images as TFRecord of Example protos.

    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      synsets: list of strings; each string is a unique WordNet ID
      labels: list of integer; each integer identifies the ground truth
      humans: list of strings; each string is a human-readable label
      bboxes: list of bounding boxes for each image. Note that each entry in this
        list might contain from 0+ entries corresponding to the number of bounding
        box annotations for the image.
      num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(synsets)
    assert len(filenames) == len(labels)
    assert len(filenames) == len(humans)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in xrange(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                synsets, labels, humans, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir, labels_file, split='train', limit=None):  # limit for dbg
    """Build a list of all images files and labels in the data set.

    Args:
      data_dir: string, path to the root directory of images.

        Assumes that the ImageNet data set resides in JPEG files located in
        the following directory structure.

          data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
          data_dir/n01440764/ILSVRC2012_val_00000543.JPEG

        where 'n01440764' is the unique synset label associated with these images.

      labels_file: string, path to the labels file.

        The list of valid labels are held in this file. Assumes that the file
        contains entries as such:
          n01440764
          n01443537
          n01484850
        where each line corresponds to a label expressed as a synset. We map
        each synset contained in the file to an integer (based on the alphabetical
        ordering) starting with the integer 1 corresponding to the synset
        contained in the first line.

        The reason we start the integer labels at 1 is to reserve label 0 as an
        unused background class.

    Returns:
      filenames: list of strings; each string is a path to an image file.
      synsets: list of strings; each string is a unique WordNet ID.
      labels: list of integer; each integer identifies the ground truth.
    """
    print('Determining list of input files and labels from %s.' % data_dir)

    challenge_synsets = []
    for l in tf.gfile.FastGFile(labels_file, 'r').readlines():
        node = data_dir + '/' + l.strip()
        if not tf.gfile.Exists(node):
            tf.logging.warn('not exsist {}'.format(node))
        else:
            challenge_synsets.append(l.strip())

        if limit is not None and len(challenge_synsets) >= limit:
            break
    if len(challenge_synsets) < limit:
        print(len(challenge_synsets))
        raise ValueError('no image')
    print('>> len of challenge synsets {}'.format(len(challenge_synsets)))
    labels = []
    filenames = []
    synsets = []

    # Leave label index 0 empty as a background class if needed
    label_index = 1

    # Construct the list of JPEG files and labels.
    for synset in challenge_synsets:
        jpeg_file_path = '%s/%s/*.JPEG' % (data_dir, synset)
        matching_files = tf.gfile.Glob(jpeg_file_path)
        # for t_ in matching_files:
        #     # print(t_)
        #
        #     utils.plt.imread(t_)
        #     from PIL import Image
        #     im = Image.open(t_)
        #     im._getexif()

        if split == 'train':
            matching_files = matching_files[:9 * len(matching_files) // 10]
        elif split == 'validation':
            matching_files = matching_files[9 * len(matching_files) // 10:]

        labels.extend([label_index] * len(matching_files))
        synsets.extend([synset] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print('Finished finding files in %d of %d classes.' % (
                label_index, len(challenge_synsets)))
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    synsets = [synsets[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.' %
          (len(filenames), len(challenge_synsets), data_dir))
    return filenames, synsets, labels


def _find_human_readable_labels(synsets, synset_to_human):
    """Build a list of human-readable labels.

    Args:
      synsets: list of strings; each string is a unique WordNet ID.
      synset_to_human: dict of synset to human labels, e.g.,
        'n02119022' --> 'red fox, Vulpes vulpes'

    Returns:
      List of human-readable strings corresponding to each synset.
    """
    humans = []
    for s in synsets:
        assert s in synset_to_human, ('Failed to find: %s' % s)
        humans.append(synset_to_human[s])
    return humans


def _process_dataset(name, directory, num_shards, synset_to_human):
    """Process a complete data set and save it as a TFRecord.

    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      synset_to_human: dict of synset to human labels, e.g.,
        'n02119022' --> 'red fox, Vulpes vulpes'
      image_to_bboxes: dictionary mapping image file names to a list of
        bounding boxes. This list contains 0+ bounding boxes.
    """
    filenames, synsets, labels = _find_image_files(directory, FLAGS.labels_file, split=name)
    humans = _find_human_readable_labels(synsets, synset_to_human)

    _process_image_files(name, filenames, synsets, labels,
                         humans, num_shards)


def _build_synset_lookup(imagenet_metadata_file):
    """Build lookup for synset to human-readable label.

    Args:
      imagenet_metadata_file: string, path to file containing mapping from
        synset to human-readable label.

        Assumes each line of the file looks like:

          n02119247    black fox
          n02119359    silver fox
          n02119477    red fox, Vulpes fulva

        where each line corresponds to a unique mapping. Note that each line is
        formatted as <synset>\t<human readable label>.

    Returns:
      Dictionary of synset to human labels, such as:
        'n02119022' --> 'red fox, Vulpes vulpes'
    """
    lines = tf.gfile.FastGFile(imagenet_metadata_file, 'r').readlines()
    synset_to_human = {}
    for l in lines:
        if l:
            parts = l.strip().split('\t')
            assert len(parts) == 2
            synset = parts[0]
            human = parts[1]
            synset_to_human[synset] = human
    return synset_to_human


def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.validation_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with '
        'FLAGS.validation_shards')
    print('Saving results to %s' % FLAGS.output_directory)

    # Build a map from synset to human-readable label.
    synset_to_human = _build_synset_lookup(FLAGS.imagenet_metadata_file)
    # Run it!

    utils.rm(FLAGS.output_directory + '/train*', block=True)
    _process_dataset('validation', FLAGS.validation_directory,
                     FLAGS.validation_shards, synset_to_human)

    # utils.rm(FLAGS.output_directory + '/validation*', block=True)
    # _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards,
    #                  synset_to_human)


if __name__ == '__main__':

    utils.init_dev(0)
    tf.app.run()