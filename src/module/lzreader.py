# encoding=utf8

'''
the reader created by luzai
'''
from __future__ import absolute_import
from utils import *
import cv2
from parrots.dnn import reader
from rediscluster import StrictRedisCluster
import random, re
import numpy as np
import sys,string

ON_DISK = "ondisk"
ON_MEM = "onmem"
REDIS = "redis"
MAX_ON_MEM = 6000000 * 20

default_conf = '''
listfile: 
    - $HOME/prj/few-shot/data/imglst/img10k.test.disk.txt
    - $HOME/prj/few-shot/data/imglst/img10k.test.redis.txt
prefix: 
    - $HOME/prj/few-shot/data/imagenet-raw
    - imagenet
    # /mnt/nfs1703/kchen/imagenet-raw-trans-to-redis
addr: 127.0.0.1:2333
delim: " "     
# shuffle: true,
# allow_io_fail: true,
# thread_num: 128,
# shuffle_epoch_num: 10
'''

tmpl = string.Template(default_conf)
default_conf= tmpl.substitute({'HOME':'/home/yxli'})
default_conf = yaml.load(default_conf)
use_pool = True

def _gget(path):
    assert os.path.exists(path)
    try:
        img_ = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
    except Exception as inst:
        print inst, 'Pls rm this img!! After one train u can disable chk', path
        rm(path, block=True)
        return None
    if img_.shape[0] <= 10 or img_.shape[1] <= 10 or img_.shape[2] <=0 or img_.shape[2]>=4:
        cv2.imwrite(randomword(10) + '.png', img_)
        print path, 'has shape: ', img_.shape, 'Pls rm!!!'
        rm(path, block=True)
        assert False
    with open(path) as fd:
        return fd.read()

class LzReader:
    support_keys = ['addr', 'listfile', 'prefix', 'delim', 'shuffle_epoch_num', 'shuffle', 'bs', 'thread_num']

    def _wrap(self, lb):
        return np.array([lb])

    def _get(self, path):
        path = self.prefix[0] + '/' + path
        try:
            if use_pool:
                self.queue.append(self.pool.apply_async(_gget, (path,)))
                if self.queue[0].ready():
                    raw = self.queue[0].get()
                    return raw
                else:
                    return None
            else:
                raw = _gget(path)
                return raw
        except KeyboardInterrupt as inst:
            print 'interrupt',inst
            self.pool.close()
            self.pool.join()
            exit(0)
        except AssertionError as inst:
            print inst, 'read {} fail'.format(path)
            self.fail_time += 1



    def _get_redis(self, key_in):
        key = self.prefix[1] + '/' + key_in
        try:
            assert self.rc.exists(key), 'radis should have {}'.format(key)
            return self.rc.get(key)
        except KeyboardInterrupt as inst:
            print 'keyin',inst
            self.pool.close()
            self.pool.join()
            exit(0)
        except AssertionError as inst:
            print inst, 'fail'
            uploadf = '/mnt/nfs1703/kchen/imagenet-raw-trans-to-redis/' + key_in
            with open(uploadf) as fd:
                upload = fd.read()
            self.rc.set('imagenet/' + key_in, upload)
            print 'upload to imagenet/' + key_in
            self.fail_time += 1


    def config(self, cfg=None):
        """
        The callback function for setting up the reader.
        New readers should implement this method.
        """
        if cfg is None:
            cfg=default_conf
            print 'debug mode!!'
        print cfg
        self.fail_time = 0
        self.th = float(cfg.get('th', 50))
        self.addr = cfg['addr'].split(":")
        self.listfile = cfg['listfile']
        self.prefix = cfg.get('prefix', "")
        self.delim = cfg.get('delim', " ,\t")
        self.shuffle_epoch_num = int(cfg.get('shuffle_epoch_num', 1))
        self.shuffle = cfg.get('shuffle', True)
        print 'init iter next'
        self.iter = self._read_iter()
        self.startup_nodes = [{"host": self.addr[0], "port": self.addr[1]}]
        self.rc = StrictRedisCluster(startup_nodes=self.startup_nodes, decode_responses=False)
        if use_pool:
            try:
                self.pool = mp.Pool(processes=128)
            except KeyboardInterrupt as inst:
                print inst
                self.pool.close()
                self.pool.join()

        self.queue = collections.deque(maxlen=128)

    def read(self):
        """
        The callback function for reading one sample.
        New readers should implement this method.
        """
        # print getattr(self,'iter',None)
        return self.iter.next()

    def _read_iter(self):
        now_on_mem = 0
        lst = read_list(self.listfile[0])
        assert len(lst)!=0
        lst = np.concatenate((np.full((lst.shape[0], 1), ON_DISK), lst, lst[:, :1]), axis=1)
        if len(self.listfile)==2:
            lst_ = read_list(self.listfile[1])
            lst_ = np.concatenate((np.full((lst_.shape[0], 1), REDIS), lst_, lst_[:, :1]), axis=1)
            lst = np.concatenate((lst, lst_), axis=0)
        lst = lst.tolist()

        while True:
            order = range(len(lst)) * self.shuffle_epoch_num
            if self.shuffle:
                random.shuffle(order)
            for i in order:
                key, img, lb, imgf = lst[i]
                if key == ON_DISK and now_on_mem < MAX_ON_MEM:
                    img = self._get(img)
                    if img is None: continue
                    lst[i] = ON_MEM, img, lb, imgf
                    now_on_mem += 1
                elif key == ON_DISK and now_on_mem >= MAX_ON_MEM:
                    img = self._get(img)
                    if img is None: continue
                elif key == REDIS:
                    img = self._get_redis(img)
                    if img is None: continue
                lb = int(lb)
                yield [img, np.array([lb])]


reader.register_pyreader(LzReader, 'lz_reader')

if __name__ == '__main__':
    lzreader = LzReader()
    lzreader.config(cfg={'listfile':['/home/yxli/prj/few-shot/data/imglst/img10k.train.disk.txt']})
    for i in range(10):
        print lzreader.read()[1]
