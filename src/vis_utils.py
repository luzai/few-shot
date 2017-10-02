from utils import  *
from graph_tool.all import *

def vis_nx(nxgraph):
    nx.write_gml(nxgraph,'tmp.gml')
    g=    load_graph('tmp.gml')
    graph_draw(g,pos=sfdp_layout(g),output='tmp.png')
    # graph_draw(g,pos=sfdp_layout(g))
