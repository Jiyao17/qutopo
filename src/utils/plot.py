
import networkx as nx
import gurobipy as gp
import matplotlib.pyplot as plt


def plot_optimized_network(graph: nx.Graph, m=None, c=None, filename: str='./result/fig.png'):
    edges = graph.edges(data=False)
    nodes = graph.nodes(data=False)

    node_colors = ['blue' if graph.nodes[node]['original'] else 'green' for node in graph.nodes]
    node_sizes = [ 200 if graph.nodes[node]['original'] else 50 for node in graph.nodes]

    # if edge channel > 0, bold the edge
    if c is not None:
        edge_widths = [ 5 if c[edge].x > 0 else 1 for edge in edges]
    # if node memory > 0, mark the node
    if m is not None:
        node_colors = ['red' if m[node].x > 0 else color for node, color in zip(nodes, node_colors)]
    # node_sizes = [ 200 if graph.nodes[node]['original'] else 50 for node in graph.nodes]
    pos = nx.get_node_attributes(graph, 'pos') 
    
    nx.draw(graph, pos, with_labels=False,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color='grey', 
        edge_cmap=plt.cm.Blues,
        width=edge_widths
        )
    
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

    pos = nx.get_node_attributes(graph, 'pos')


    node_labels = nx.get_node_attributes(graph, node_label)
    edge_labels = nx.get_edge_attributes(graph, edge_label)
    node_colors = ['blue' if graph.nodes[node]['original'] else 'green' for node in graph.nodes]
    # node_colors = ['blue' for node in graph.nodes]
    node_sizes = [ 200 if graph.nodes[node]['original'] else 50 for node in graph.nodes]

    nx.draw(graph, pos, with_labels=False, 
        node_color=node_colors, 
        node_size=node_sizes,
        edge_color='grey', width=1, edge_cmap=plt.cm.Blues
        )
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.savefig(filename)
    plt.close()

