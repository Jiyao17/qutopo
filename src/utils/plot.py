
import networkx as nx
import gurobipy as gp
import matplotlib.pyplot as plt

from ..network import FusedTopo


def plot_opt_result(topo: FusedTopo, solver: gp.Model=None, filename: str='./result/fig.png'):
    # set edge channel number to optimization result
    for edge in topo.net.edges:
        if solver is not None:
            topo.net.edges[edge]['channels'] = solver.c[edge].x
        else:
            topo.net.edges[edge]['channels'] = 1
    # set node capacity to optimization result
    for node in topo.net.nodes:
        if solver is not None:
            topo.net.nodes[node]['capacity'] = solver.m[node].x
        else:
            topo.net.nodes[node]['capacity'] = 1
    # draw the multigraph, save the result to a file
    # set capacity as the edge and node labels
    pos = nx.spring_layout(topo.net)
    edge_labels = {
        (u, v): str(int(d['channels'])) + '-' + str(int(d['length'])) + '-' + '%.2f' % d['prob']
                                for u, v, k, d in topo.net.edges(data=True, keys=True)
        }
    node_labels = {node: int(topo.net.nodes[node]['capacity']) for node in topo.net.nodes}
    nx.draw(topo.net, pos, with_labels=False, node_color='lightblue', node_size=500, edge_color='grey', width=1, edge_cmap=plt.cm.Blues)
    nx.draw_networkx_edge_labels(topo.net, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(topo.net, pos, labels=node_labels)

    plt.savefig(filename)


