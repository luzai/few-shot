import utils
import tensorflow as tf
from utils import *
import cv2


def check_individual(filename, filepath):
    if filename.endswith('JPEG') and osp.getsize(filepath) == 0:
        utils.rm(filepath)
        append_file(filepath, file='/home/wangxinglu/prj/few-shot/src/append.txt')
        return

    im = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_COLOR)
    cv2.imwrite(filepath, im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if im.shape[0] <= 10 or im.shape[1] <= 10:
        # utils.rm(filepath)
        append_file(filepath, './wrong.img.txt')
        print filepath
    assert im.shape[-1] == 3
    assert im.shape[0] > 13 and im.shape[1] > 13
    assert (len(im.shape) == 3 and im.shape[-1] == 3), 'img ' + filepath + str(im.shape)
    assert im.shape[1] != 0 and im.shape[0] != 0, 'width ' + filepath + str(im.shape)


@chdir_to_root
def check_img(prefix='/home/wangxinglu/prj/few-shot/data/imagenet-raw'):
    import multiprocessing as mp
    pool = mp.Pool(1024)
    os.chdir(prefix)
    for dirpath, dirnames, filenames in tf.gfile.Walk('.'):
        for filename in filenames:
            if filename.endswith('JPEG'):
                filepath = dirpath + '/' + filename
                pool.apply_async(check_individual, args=(filename, filepath))
                # check_individual(filename, filepath)


@chdir_to_root
def chk_img_lst(path='/home/wangxinglu/prj/few-shot/data/imglst/img10k.test.txt'):
    import multiprocessing as mp
    pool = mp.Pool(10242)
    os.chdir('/home/wangxinglu/prj/few-shot/data/imagenet-raw')
    print np.array(read_list(path)),'ok'
    for imgpath in np.array(read_list(path))[:, 0]:
        pool.apply_async(check_individual, args=(imgpath.split('/')[-1], imgpath))
    pool.close()
    pool.join()


if __name__ == '__main__':
    chk_img_lst()
    # chk_img_lst(path='/home/wangxinglu/prj/few-shot/data/imglst/img10k.train.txt')
    # check_img()
    # check_img(prefix='/mnt/nfs1703/kchen/imagenet-raw-trans-to-redis')
