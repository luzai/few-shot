# -*- coding: utf-8 -*-
import multiprocessing as mp
import subprocess, sys, os
from xml.etree import ElementTree
import requests
from utils import *
from metadata import *
import sys


def test_url(url, dst, params={}):
    r = requests.head(url, params=params)
    content_type = r.headers["content-type"]
    if content_type.startswith("text"):
        print r.url
        print r.status_code
        print TypeError("404 Error"), '\n'
        return
    else:
        print 'ok', r.headers["content-type"]


def download_file(url, dst, params={}, debug=True):
    # if dst in read_list('./404.txt'):
    #     return
    if debug:
        print u"downloading {0}...".format(dst),
    # from IPython import embed;embed()
    response = requests.get(url, params=params)
    content_type = response.headers["content-type"]
    if content_type.startswith("text"):
        print response.url
        print response.status_code
        print TypeError("404 Error"), '\n'
        append_file(dst, './404.txt')
        return
    else:
        with file(dst, "wb") as fp:
            fp.write(response.content)
        if osp.getsize(dst) == 0 or osp.getsize(dst) == 10240:
            append_file(dst, './404.txt')
        print 'ok'
        print response.url, "done.\n"
        # mkdir_p(dst.rstrip('.tar'), delete=True)
        # tar(dst, dst.rstrip('.tar'))
        # rm(dst, block=True)
        # while os.path.exists(dst): pass
        # if 'SSD' in dst:
        #     from metadata import config
        #     ln(dst.rstrip('.tar'), config.filepath+'/')
        #     ln(dst.rstrip('.tar'), config.filepath+'2/')


def find_path(folder):
    import os
    res = []
    for root, dirs, files in os.walk('.'):
        for dir in dirs:
            if dir == folder:
                res.append(os.path.join(root, dir))
    return res


def travel_tree():
    with file(config.structure_released, "r") as fp:
        tree = ElementTree.parse(fp)
        root = tree.getroot()
        release_data = root[0].text
        synsets = root[1]

        # from IPython import embed;embed()
        for child in synsets.iter():
            if len(child) > 0:
                continue
            yield child


if __name__ == "__main__":
    pools = mp.Pool(processes=64)
    ttl_category = 0
    task_l = []
    nodes = read_list('/home/wangxinglu/prj/few-shot/src/corrupt')
    # nodes = np.unique(nodes)
    nodes = set()
    nodes = nodes.union(set(imagenet1k))
    nodes = nodes.union(set(imagenet7k))
    nodes = nodes.union(set(shuffle_iter(nx.dfs_preorder_nodes(ori_tree, 'fall11'))))
    for node in nodes:
        if len(ori_tree.successors(node)) > 0:
            continue
        ttl_category += 1
        wnid = node

        # todo I cannot wait network, so put it to precious ssd space now

        imagepath_ssd = get_imagepath(wnid, ssd=True)
        imagepath = get_imagepath(wnid)

        if osp.exists(imagepath_ssd) and osp.getsize(imagepath_ssd) != 0:
            continue
        if osp.exists(imagepath) and osp.getsize(imagepath) != 0:
            continue
        if osp.exists(imagepath_ssd.strip('.tar')) and len(os.listdir(imagepath_ssd.strip('.tar'))) != 0:
            continue
        if osp.exists(imagepath.strip('.tar')) and len(os.listdir(imagepath.strip('.tar'))) != 0:
            continue

        params = {
            "wnid": wnid,
            "username": config.username,
            "accesskey": config.accesskey,
            "release": "latest",
            "src": "stanford"
        }
        try:
            task_l.append(pools.apply_async(download_file, (config.synset_url, imagepath_ssd, params)))
            # download_file(config.synset_url, imagepath_ssd, params)
        except Exception as inst:
            print inst
    for task in task_l:
        try:
            task.get()
        except Exception as inst:
            print inst

    print ttl_category
