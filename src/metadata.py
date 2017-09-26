from xml.etree import ElementTree
from utils import *



def _read(file, delimiter=None):
    mapping_ = np.genfromtxt(file, delimiter=delimiter, dtype='str')
    mapping = {}

    for two in mapping_:
        mapping[two[0]] = two[1]
    return mapping


is_a = _read(root_path + '/data/wordnet.is_a.txt')
id2word = _read(root_path + '/data/words.txt', delimiter='\t')


def _read_list(file):
    lines = np.genfromtxt(file, dtype='str')
    return lines


imagenet10k = _read_list(root_path + '/data/imagenet10k.txt')
imagenet22k = _read_list(root_path + '/data/imagenet22k.txt')
imagenet7k = _read_list(root_path + '/data/imagenet7k.txt')
imagenet1k = _read_list(root_path + '/data/imagenet1k.txt')
imagenet600 = _read_list(root_path + '/data/imagenet600.txt')

import easydict

config = easydict.EasyDict()

config.synset_url = u"http://www.image-net.org/download/synset"
config.username = u"luzai"
config.accesskey = u"a1b86dba55cbb6cc765268b6cf186284ed793a32"
config.filepath = "/mnt/nfs1703/kchen/imagenet-raw"  # root_path + "/data/imagenet22k-raw"
config.filepath_ssd = "/mnt/SSD/luzai/imagenet22k-raw"

config.base_url = u"http://www.image-net.org/api/xml/"
config.structure_released = root_path + "/data/structure_released.xml"

graph = nx.DiGraph()



with file(config.structure_released, "r") as fp:
    tree_xml = ElementTree.parse(fp)
    root = tree_xml.getroot()
    release_data = root[0].text
    synsets = root[1]

    # from IPython import embed;embed()
    for child in synsets.iter():
        wnid = child.attrib.get("wnid")
        for cd in child:
            wd = cd.attrib.get("wnid")
            graph.add_edge(wnid, wd)

    print graph

tree = nx.bfs_tree(graph, 'fall11')


@chdir_to_root
def get_imagepath(wnid, ssd=False):
    return os.path.join(config.filepath, wnid + ".tar") if not ssd else (config.filepath_ssd + '/' + wnid + '.tar')


@chdir_to_root
def construct_path():
    os.chdir('./data')
    # mkdir_p("fall11", delete=True)

    for node in nx.dfs_preorder_nodes(tree, 'fall11'):
        hist = nx.shortest_path(tree, "fall11", node)
        path = list2str(hist, delimier='/')
        imagepath = get_imagepath(node)
        if os.path.exists(imagepath) and os.path.getsize(imagepath) != 0:
            # mkdir_p(path)
            # mkdir_p(imagepath.strip('.tar'))
            tar(imagepath, imagepath.strip('.tar'))
            # rm(path)
            # ln(imagepath.strip('.tar'), path.strip('/'))
            # print path, imagepath.strip('.tar')


@chdir_to_root
def construct_path_from(tree_, res):
    os.chdir('./data')
    path = 'train'
    mkdir_p(path)
    os.chdir(path)
    for node in res:
        if tree_.node[node]['nchild'] == 0:
            nodes = [node]
        else:
            nodes = [node_ for node_ in nx.dfs_preorder_nodes(tree_, node) if tree_.node[node_]['nchild'] == 0]
        for node_ in nodes:
            imagepath = get_imagepath(node_).strip('.tar')
            mkdir_p(node)
            ln(imagepath + '/*', node)


@chdir_to_root
def nimg_per_class(tree_):
    all_nimg = []
    for node in nx.dfs_preorder_nodes(tree_, 'fall11'):
        imagepath = get_imagepath(node).strip('.tar')
        if osp.exists(imagepath) and tree_.node[node]['nchild'] == 0:
            nimg = len(os.listdir(imagepath))
            all_nimg.append(nimg)
            if nimg == 0:
                print node
                rm(imagepath)
    return np.sort(all_nimg)[::-1]


@chdir_to_root
def tag_tree(tree_):
    max_depth = 0
    all_depth = []
    all_nchild = []
    # all_nimg = []
    for node in nx.dfs_preorder_nodes(tree_, 'fall11'):
        depth = nx.shortest_path_length(tree_, "fall11", node)
        max_depth = max(depth, max_depth)
        all_depth.append(depth)
        nchild = len(tree_.successors(node))
        all_nchild.append(nchild)
        # if not nchild == 0:
        tree_.add_node(node, depth=depth, nchild=nchild)
    tree_.max_depth = max_depth
    tree_.all_depth = all_depth
    tree_.all_nchild = all_nchild
    return tree_


@chdir_to_root
def slim_tree(tree_, condition=None):
    '''
    slim tree, remove node that not in imagenet7k and those download fail(may be exists)
    :param tree_:
    :return:
    '''
    os.chdir('data')

    new_tree = nx.DiGraph()
    if isinstance(condition,basestring):
        condition=eval(condition)

    for node in nx.dfs_preorder_nodes(tree_, 'fall11'):
        hist = nx.shortest_path(tree_, "fall11", node)
        imagepath = get_imagepath(node)
        imagepath = imagepath.strip('.tar')
        if tree_.node[node]['nchild'] == 0:
            if condition is not None and node in condition:
                new_tree.add_path(hist)
            elif condition is None:
                new_tree.add_path(hist)
    new_tree = tag_tree(new_tree)
    return new_tree


@chdir_to_root
def vis_tree(tree_):
    all_nodes = ['fall11', ]
    for t_ in nx.algorithms.bfs_successors(tree_, 'fall11').values():
        all_nodes.extend(t_)
    all_nodes = np.unique(all_nodes)

    nodes = all_nodes[:14]
    vis_nx(nx.subgraph(tree_, nodes))


@chdir_to_root
def dir2tree():
    os.chdir('./data')
    tree_ = nx.DiGraph()
    i = 0
    for root, dirs, files in os.walk('./fall11'):
        i += 1

        # print root.split('/')[-1], dirs, files
        for dir in dirs:
            tree_.add_edge(root.split('/')[-1], dir)
            # if i > 3:
            #     break
    # tree_
    return tree_


tree = ori_tree = tag_tree(tree)

# tree = ori_tree
# tree2=slim_tree(tree,condition='imagenet1k')
# pass