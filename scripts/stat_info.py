%pwd
from model_utils import *
from metadata import *
from gen_imglst import *
fn='/mnt/nfs1703/test/prj/few-shot/data/imagenet10k.txt'
fn='/mnt/nfs1703/test/prj/few-shot/data/imagenet10k.no1k'
l = read_list(fn)
l[:10]


def stat_info(imagenet1k):
    nimgs=[]
    for node in imagenet1k:
        imagepath = get_imagepath(node).strip('.tar')
        if osp.exists(imagepath):
            nimgs.append(len(os.listdir(imagepath)))
        else:
            nimg=0
            for node_ in find_child(ori_tree,node):
                imagepath=get_imagepath(node_).strip('.tar')
                nimg+=len(os.listdir(imagepath))
            if nimg ==0:
#                 print node_
                pass
            else:
                nimgs.append(nimg)

    nimgs = np.sort(nimgs)[::-1]
    print sum(nimgs),sum(nimgs)/len(nimgs),len(nimgs)

    fig,axes=plt.subplots(ncols=2,nrows=1,figsize=(9,2))
    _=axes[0].hist(nimgs)
    axes[1].plot(nimgs[nimgs<3048])
#     plt.yscale('log')
    return nimgs
stat_info(l)
