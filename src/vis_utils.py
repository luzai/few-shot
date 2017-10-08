from utils import *
from graph_tool.all import *


def vis_nx(nxgraph,tree=False):
    nx.write_gml(nxgraph, 'tmp.gml')
    g = load_graph('tmp.gml')
    if tree:
        graph_draw(g,pos=radial_tree_layout(g,'fall11'),output='tmp.png')
    else:
        graph_draw(g, pos=sfdp_layout(g), output='tmp.png')

    # graph_draw(g,pos=sfdp_layout(g))
