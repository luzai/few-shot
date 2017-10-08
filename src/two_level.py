from utils import *
import utils


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


import matplotlib

matplotlib.style.use('ggplot')

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

nimgs = unpickle('d.pkl')
nimgs = {k: len(v) for k, v in nimgs.iteritems()}
tree = nx.read_gml('tree.gml')

low = 2500
up = 7500
visited = set()
coarse = []
fine = []
infos = []
sav = []
for node in nx.dfs_preorder_nodes(tree, 'fall11'):
    if node in visited: continue
    visited.add(node)
    depth = tree.node[node]['depth']
    nchild = tree.node[node]['nchild']
    if len(tree.successors(node)) == 0:
        nimg = (nimgs[node])
        imgs = [node]
    else:
        imgs = find_leaves(tree, node, chk=False)
        nimg = sum([nimgs[n_] for n_ in imgs])
    if nimg < low:
        if nchild == 0:
            infos.append([node, depth, nchild, nimg])
    elif nimg < up:
        sav.append(nimg)
        visited = visited.union(find_child_recursive(tree, node))
        coarse.append(node)
        fine.append(imgs)

df = pd.DataFrame.from_records(infos, columns=['node', 'depth', 'nchild', 'nimg'])
print len(fine)
for t in df.sort_values(by='depth').groupby('depth'):
    print len(fine)
    df_t = t[1].iloc[np.random.permutation(t[1].shape[0]), :]
    coarset = df_t['node'].iloc[0]
    finet = [df_t['node'].iloc[0]]
    sumt = df_t['nimg'].iloc[0]
    for i_ in range(1,df_t.shape[0]):
        if sumt<low:
            sumt+=df_t['nimg'].iloc[i_]
            finet.append(df_t['node'].iloc[i_])

        else:
            sav.append(sumt)
            sumt=df_t['nimg'].iloc[i_]
            coarse.append(coarset)
            fine.append(finet)

            coarset=df_t['node'].iloc[i_]
            finet = [df_t['node'].iloc[i_]]
    sav.append(sumt)
    sumt=0
    coarse.append(coarset)
    fine.append(finet)