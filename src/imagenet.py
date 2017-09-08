# -*- coding: utf-8 -*-

import subprocess
import sys
import os
from xml.etree import ElementTree
import requests
from utils import *
# from vis_utils import *
# from plt_utils import *
# from model_utils import *

import sys
import os


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
    if debug:
        print u"downloading {0}...".format(dst),
    # from IPython import embed;embed()
    response = requests.get(url, params=params)
    content_type = response.headers["content-type"]
    if content_type.startswith("text"):
        print response.url
        print response.status_code
        print TypeError("404 Error"), '\n'
        return
    else:
        with file(dst, "wb") as fp:
            fp.write(response.content)
        print 'ok'
        print response.url, "done.\n"


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


from metadata import *

if __name__ == "__main__":
    i = 0

    for node in shuffle_iter(nx.dfs_preorder_nodes(tree, 'fall11')):
        # for node in nx.dfs_preorder_nodes(tree, 'fall11'):
        if len(tree.successors(node)) > 0:
            continue
        i += 1
        wnid = node

        imagepath = get_imagepath(wnid)
        print imagepath
        if not node in imagenet10k:
            print node, 'not in imagenet10k'
            continue
        if osp.exists(imagepath) and osp.getsize(imagepath) != 0:
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
        download_file(config.synset_url, imagepath, params)
    print i
