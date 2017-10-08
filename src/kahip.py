from utils import *
import utils
import matplotlib

matplotlib.style.use('ggplot')


def find_leaves(tree_, node, chk=True):
    res = []
    try:
        for node in nx.dfs_preorder_nodes(tree_, node):
            if tree_.node[node]['nchild'] != 0: continue
            res.append(node)
    except Exception as inst:
        print inst, 'wrong'
    return res


def find_child_recursive(tree_, node):
    res = set()
    for node in nx.dfs_preorder_nodes(tree_, node):
        res.add(node)
    return res


nimgs = unpickle('d.pkl')
nimgs = {k: len(v) for k, v in nimgs.iteritems()}
tree = nx.read_gml('tree.gml')
childs = find_leaves(tree, 'fall11')


def tag_tree(tree_):
    max_depth = 0
    all_depth = []
    all_nchild = []
    for node in nx.dfs_preorder_nodes(tree_, 'fall11'):
        depth = nx.shortest_path_length(tree_, "fall11", node)
        max_depth = max(depth, max_depth)
        all_depth.append(depth)
        nchild = len(list((tree_.successors(node))))
        all_nchild.append(nchild)

        height = max([nx.shortest_path(tree_, node, t) for t in find_leaves(tree_, node)])

        tree_.add_node(node, depth=depth, nchild=nchild, height=height)
    tree_.max_depth = max_depth
    tree_.all_depth = all_depth
    tree_.all_nchild = all_nchild
    return tree_


tree = tag_tree(tree)


def edge_weight(tree, ni, nj):
    p1 = nx.shortest_path(tree, 'fall11', ni)
    p2 = nx.shortest_path(tree, 'fall11', nj)
    for ind, (t1, t2) in enumerate(zip(p1, p2)):
        if t1 != t2: break
    ind -= 1
    node = p1[ind]
    return tree.node[node]['height']


v = np.array(childs)
vl = v.shape[0]

w = [[edge_weight(tree, v[i], v[j]) for i in range(vl)] for j in range(vl)]
w = np.array(w)

with open('res.txt', 'w') as f:
    f.write('{} {} {}\n'.format(v.shape[0], (vl * vl - vl) // 2, 11))
    # for
