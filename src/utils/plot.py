

from copy import deepcopy

import networkx as nx
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np


def plot_optimized_network(graph: nx.Graph, m=None, c=None, phi=None, filename: str='./result/fig.png'):
    graph = deepcopy(graph)
    
    empty_nodes = [ node for node in m if m[node].x == 0]
    graph.remove_nodes_from(empty_nodes)
    nodes = graph.nodes(data=False)
    edges = graph.edges(data=False)

    node_colors = ['blue' if graph.nodes[node]['group'] == 0 else 'green' for node in graph.nodes]
    node_sizes = [ 200 if graph.nodes[node]['group'] == 0 else 50 for node in graph.nodes]

    # if edge channel > 0, bold the edge
    if c is not None:
        edge_widths = [ 5 if c[edge].x > 0 else 1 for edge in edges]
        edge_labels = {}
        for edge in edges:
            ch = int(c[edge].x)
            p = int(np.ceil(phi[edge].x))
            if ch > 0:
                edge_labels[edge] = f'{ch}-{p}'
                # edge_labels[edge] = f'{p}'
            else:
                edge_labels[edge] = ''
    # if node memory > 0, mark the node
    if m is not None:
        node_colors = ['red' if m[node].x > 0 else color for node, color in zip(nodes, node_colors)]
        node_labels = {node: int(m[node].x) if m[node].x > 0 else '' for node in nodes}
    # node_sizes = [ 200 if graph.nodes[node]['original'] else 50 for node in graph.nodes]
    pos: dict = nx.get_node_attributes(graph, 'pos') 
    pos = {node: (lon, lat) for node, (lat, lon) in pos.items()}
    
    nx.draw(graph, pos, with_labels=False,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color='grey', width=edge_widths, edge_cmap=plt.cm.Blues
        )
    
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    
    plt.savefig(filename)
    plt.close()



def plot_nx_graph(
        graph: nx.Graph,
        node_label: str='id',
        edge_label: str='length',
        filename: str='./result/fig.png'
        ):
    """
    plot the network
    nodes layout is based on coordinates
    """

    pos: dict = nx.get_node_attributes(graph, 'pos')
    # reverse latitude and longitude
    pos = {node: (lon, lat) for node, (lat, lon) in pos.items()}


    node_labels = nx.get_node_attributes(graph, node_label)
    edge_labels = nx.get_edge_attributes(graph, edge_label)
    node_colors = ['blue' if graph.nodes[node]['group'] == 0 else 'green' for node in graph.nodes]
    # node_colors = ['blue' for node in graph.nodes]
    node_sizes = [ 200 if graph.nodes[node]['group'] == 0 else 50 for node in graph.nodes]

    nx.draw(graph, pos, with_labels=False, 
        node_color=node_colors, 
        node_size=node_sizes,
        edge_color='grey', width=1, edge_cmap=plt.cm.Blues
        )
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.savefig(filename)
    plt.close()

