import utils
import tensorflow as tf
from utils import *


def check_img():
    prefix = '/home/wangxinglu/prj/few-shot/data/imagenet-raw'
    os.chdir(prefix)
    for dirpath, dirnames, filenames in tf.gfile.Walk('.'):
        for filename in filenames:
            filepath = dirpath + '/' + filename
            if filename.endswith('JPEG') and osp.getsize(filepath) == 0:
                utils.rm(filepath)
                append_file(filepath, file='/home/wangxinglu/prj/few-shot/src/append.txt')
