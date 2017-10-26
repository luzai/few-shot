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
    all_height = []
    for node in nx.dfs_preorder_nodes(tree_, 'fall11'):
        depth = nx.shortest_path_length(tree_, "fall11", node)
        max_depth = max(depth, max_depth)
        all_depth.append(depth)
        nchild = len(list((tree_.successors(node))))
        all_nchild.append(nchild)

        height = max([nx.shortest_path_length(tree_, node, t) for t in find_leaves(tree_, node)])
        all_height.append(height)
        tree_.add_node(node, depth=depth, nchild=nchild, height=height)
    tree_.max_depth = max_depth
    tree_.all_depth = all_depth
    tree_.all_nchild = all_nchild
    tree_.all_height = all_height
    return tree_


tree = tag_tree(tree)


def edge_weight(inp):
    ni, nj = inp
    p1 = nx.shortest_path(tree, 'fall11', ni)
    p2 = nx.shortest_path(tree, 'fall11', nj)
    assert p1[0] == 'fall11'
    for ind, (t1, t2) in enumerate(zip(p1, p2)):
        if t1 != t2: break
    ind -= 1
    node = p1[ind]
    return tree.node[node]['height']

v = np.array(childs)

# ch=np.random.choice(range(len(v)),(3,))
# ch=np.sort(ch)
# ch = np.arange(9998)
# v = v[ch]
vl = v.shape[0]

# inp = []
# w = np.zeros(shape=(vl, vl), dtype=int)
# for i in range(vl):
#     for j in range(i + 1, vl):
#         # inp.append([v[i], v[j]])
#         w[i, j] = edge_weight((v[i], v[j]))
#         # break
# # pool = mp.Pool(128)
# # wt = pool.map(edge_weight, inp)
# pickle(w,'tmp.pkl')
w=unpickle('tmp.pkl')
# w=w[ch,:][:,ch]
w=w.T+w
