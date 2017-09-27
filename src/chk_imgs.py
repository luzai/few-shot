import utils
import tensorflow as tf
from utils import *
import cv2


def check_individual(filename, filepath):
    if filename.endswith('JPEG') and osp.getsize(filepath) == 0:
        utils.rm(filepath)
        append_file(filepath, file='/home/wangxinglu/prj/few-shot/src/append.txt')
        return
    try:
        im = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_COLOR)
        # cv2.imwrite(filepath, im, 100)
        assert (len(im.shape) == 3 and im.shape[-1] == 3), 'img ' + filepath + str(im.shape)
        assert im.shape[1] != 0 and im.shape[0] != 0, 'width ' + filepath + str(im.shape)

    except Exception as inst:
        print inst, filepath
        # pass


def check_img():
    import multiprocessing as mp
    pool = mp.Pool(1024)
    prefix = '/home/wangxinglu/prj/few-shot/data/imagenet-raw'
    os.chdir(prefix)
    for dirpath, dirnames, filenames in tf.gfile.Walk('.'):
        for filename in filenames:
            filepath = dirpath + '/' + filename
            pool.apply_async(check_individual, args=(filename, filepath))
            # check_individual(filename, filepath)


if __name__ == '__main__':
    check_img()