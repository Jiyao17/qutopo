
import networkx as nx
import gurobipy as gp
import matplotlib.pyplot as plt



def plot_opt_result(topo, solver: gp.Model=None, filename: str='./result/fig.png'):
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
        (u, v): str(int(d['channels'])) + '-' + str(int(d['length'])) + '-' + '%.2f' % d['cap_per_channel']
                                for u, v, k, d in topo.net.edges(data=True, keys=True)
        }
    node_labels = {node: int(topo.net.nodes[node]['capacity']) for node in topo.net.nodes}
    nx.draw(topo.net, pos, with_labels=False, node_color='lightblue', node_size=500, edge_color='grey', width=1, edge_cmap=plt.cm.Blues)
    nx.draw_networkx_edge_labels(topo.net, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(topo.net, pos, labels=node_labels)

    plt.savefig(filename)
    plt.close()


def plot_nx_graph(
        graph: nx.Graph,
        labeled_edges: bool=False,
        filename: str='./result/fig.png'
        ):
    """
    plot the network
    """
    pos = nx.spring_layout(graph)
    # length as edge labels, if exists
    if labeled_edges and graph.number_of_edges() > 0:
        edge_data: dict = list(graph.edges(data=True))[0][2]
        if 'length' in edge_data:
            edge_labels: dict = nx.get_edge_attributes(graph, 'length')
            #round the length to 2 decimal places
            edge_labels = {key: '%.2f' % value for key, value in edge_labels.items()}
    else:
        edge_labels = None
    nx.draw(
        graph, pos, with_labels=False, 
        node_color='lightblue', node_size=200, 
        edge_color='grey', width=1, edge_cmap=plt.cm.Blues
        )
    if edge_labels is not None:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    plt.savefig(filename)
    plt.close()

