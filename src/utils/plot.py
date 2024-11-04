

from copy import deepcopy

import networkx as nx
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np


def plot_optimized_network(graph: nx.Graph, m=None, c=None, phi=None, labeled=False, filename: str='./result/fig.png'):
    """
    m: dict[int, int]
        - memory allocation
    c: dict[tuple[int], int]
        - channel allocation
    phi: dict[tuple[int], float]
        - channel utilization
    """
    
    graph = deepcopy(graph)
    
    empty_nodes = [ node for node in m if m[node] == 0]
    graph.remove_nodes_from(empty_nodes)
    nodes = graph.nodes(data=False)
    edges = graph.edges(data=False)

    node_colors = ['blue' if graph.nodes[node]['group'] == 0 else 'green' for node in graph.nodes]
    node_sizes = [ 200 if graph.nodes[node]['group'] == 0 else 50 for node in graph.nodes]

    # if edge channel > 0, bold the edge
    if c is not None:
        edge_widths = [ 5 if c[edge] > 0 else 1 for edge in edges]
        edge_labels = {}
        for edge in edges:
            ch = c[edge]
            p = int(np.ceil(phi[edge]))
            if ch > 0:
                edge_labels[edge] = f'{ch}-{p}'
                # edge_labels[edge] = f'{p}'
            else:
                edge_labels[edge] = ''
    # if node memory > 0, mark the node
    if m is not None:
        node_colors = ['red' if m[node] > 0 else color for node, color in zip(nodes, node_colors)]
        node_labels = {node: m[node] if m[node] > 0 else '' for node in nodes}
    # node_sizes = [ 200 if graph.nodes[node]['original'] else 50 for node in graph.nodes]
    pos: dict = nx.get_node_attributes(graph, 'pos') 
    pos = {node: (lon, lat) for node, (lat, lon) in pos.items()}
    
    nx.draw(graph, pos, with_labels=False,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color='grey', width=edge_widths, edge_cmap=plt.cm.Blues
        )
    
    if labeled:
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
    graph = deepcopy(graph)
    pos: dict = nx.get_node_attributes(graph, 'pos')
    # reverse latitude and longitude
    # for node in pos:
    #     lat, lon = pos[node]
    #     if not 0 < lat < 90 or not -180 < lon < 180:
    #         raise ValueError(f'Invalid latitude and longitude: {lat}, {lon}')
    pos = {node: (lon, lat) for node, (lat, lon) in pos.items()}



    node_labels = nx.get_node_attributes(graph, node_label)
    edge_labels = nx.get_edge_attributes(graph, edge_label)
    node_colors = ['blue' if graph.nodes[node]['group'] == 0 else 'red' for node in graph.nodes]
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




def plot_simple_topology(
        graph: nx.Graph,
        Im=None, Ic=None,
        users=None,
        filename: str='./result/fig.png'
        ):
    """
    plot the network
    """
    graph = deepcopy(graph)
    pos: dict = nx.get_node_attributes(graph, 'pos')
    # reverse latitude and longitude
    # for node in pos:
    #     lat, lon = pos[node]
    #     if not 0 < lat < 90 or not -180 < lon < 180:
    #         raise ValueError(f'Invalid latitude and longitude: {lat}, {lon}')
    pos = {node: (lon, lat) for node, (lat, lon) in pos.items()}
    # node_colors = ['blue' if graph.nodes[node]['group'] == 0 else 'green' for node in graph.nodes(data=False)]
    # node_colors = ['blue' for node in graph.nodes]
    node_sizes = [ 200 if graph.nodes[node]['group'] == 0 else 50 for node in graph.nodes]


    if Im is None:
        Im = {node: 1 for node in graph.nodes(data=False)}
    if Ic is None:
        Ic = {edge: 1 for edge in graph.edges(data=False)}
    if users is None:
        users = graph.nodes(data=False)

    # node_colors = ['blue' if graph.nodes[node]['group'] == 0 else 'red' for node in graph.nodes]
    # node_colors = ['blue' if Im[node] == 1 else 'red' for node in Im.keys()]
    # for user in users:
    #     node_colors[user] = 'green'

    # remove unused nodes
    empty_nodes = [ node for node in Im.keys() if Im[node] == 0]
    graph.remove_nodes_from(empty_nodes)
    # node_colors = [color for node, color in zip(graph.nodes, node_colors)]
    # node_sizes = [size for node, size in zip(graph.nodes, node_sizes)]
    # remove all edges
    graph.remove_edges_from(list(graph.edges))
    # add edges with channel utilization
    for edge in Ic:
        u, v = edge
        if Ic[edge] > 0:
            graph.add_edge(u, v, length=Ic[edge])

    node_sizes = [ 200 if graph.nodes[node]['group'] == 0 else 50 for node in graph.nodes]
    node_colors = ['green' if size == 200 else 'blue' for size in node_sizes]
    nx.draw(graph, pos, with_labels=False, 
        node_color=node_colors, 
        node_size=node_sizes,
        edge_color='grey', width=1, edge_cmap=plt.cm.Blues
    )

    plt.savefig(filename)
    plt.close()



def plot_lines(
    x, ys,
    xlabel, ylabel,
    labels, 
    colors=None, 
    markers=None,
    adjust=(0.2, 0.2, 0.95, 0.95),
    xscale='linear', yscale='linear',
    xticklabel=None, yticklabel=None,
    xreverse=False, yreverse=False,
    xlim=None, ylim=None,
    filename='pic.png',
    ):
    plt.figure()
    plt.rc('font', size=20)
    plt.subplots_adjust(*adjust)

    if colors is None:
        # get matpliotlib default color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if markers is None:
        markers = ['o', 's', '^', 'v', 'x', '+', 'd', 'p', 'h']
    
    for y, label, color, marker in zip(ys, labels, colors, markers):
        # find first y > ylim[1]
        if ylim is not None:
            idx = np.where(y > ylim[1])[0]
            if len(idx) > 0:
                y = y[:idx[0]]
                x = x[:idx[0]]
        # hollow markers
        plt.plot(x, y, label=label, color=color, marker=marker, markerfacecolor='none', markersize=10)

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xscale(xscale)
    plt.yscale(yscale)

    if xticklabel is not None:
        plt.ticklabel_format(axis='x', style=xticklabel, scilimits=(0,0))
    # if yticklabel is not None:
    #     plt.ticklabel_format(axis='y', style=yticklabel, scilimits=(0,0))
    if xreverse:
        plt.gca().invert_xaxis()
    if yreverse:
        plt.gca().invert_yaxis()
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    # plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)

    plt.close()


def plot_2y_lines(
    x, ys1, ys2,
    xlabel, y1_label, y2_label,
    y1_labels, y2_labels,
    y1_styles, y2_styles,
    y1_markers, y2_markers,
    y1_colors=None, y2_colors=None,
    xscale='linear', y1_scale='linear', y2_scale='linear',
    x_tickstyle=None, y1_tickstyle=None, y2_tickstyle=None,
    xreverse=False, y1_reverse=False, y2_reverse=False,
    xlim=None, y1_lim=None, y2_lim=None,
    filename='pic.png',
    ):
    """
    two y-axis in one figure
    """
    if y1_colors is None:
        y1_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if y2_colors is None:
        y2_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if y1_labels is None:
        y1_labels = ['' for _ in ys1]
    if y2_labels is None:
        y2_labels = ['' for _ in ys2]
    fig, ax1 = plt.subplots()
    plt.rc('font', size=16)
    plt.subplots_adjust(0.18, 0.16, 0.96, 0.96)
    ax2 = ax1.twinx()

    for y1, label, style, color, marker in zip(ys1, y1_labels, y1_styles, y1_colors, y1_markers):
        # find first y > ylim[1]
        x1 = x
        # cut off the tail
        if y1_lim is not None:
            idx = np.where(y1 > y1_lim[1])[0]
            if len(idx) > 0:
                y1 = y1[:idx[0]]
                x1 = x[:idx[0]]
        ax1.plot(
            x1, y1, 
            label=label, linestyle=style, color=color, marker=marker,
            markerfacecolor='none', markersize=10
            )
    
    for y2, label, style, color, marker in zip(ys2, y2_labels, y2_styles, y2_colors, y2_markers):
        # cut off the tail
        x2 = x
        if y2_lim is not None:
            idx = np.where(y2 > y2_lim[1])[0]
            if len(idx) > 0:
                y2 = y2[:idx[0]]
                x2 = x[:idx[0]]
        ax2.plot(x2, y2, 
            label=label, linestyle=style, color=color, marker=marker,
            markerfacecolor='none', markersize=10
            )
        
    # force x ticks to be integers and multiple of 5, including 0
    # ax1.set_xticks(np.arange(0, x[-1], 5))
    ax1.set_xlabel(xlabel, fontsize=20)
    ax1.set_ylabel(y1_label, fontsize=20)
    ax2.set_ylabel(y2_label, fontsize=20)
    ax1.set_xscale(xscale)
    ax1.set_yscale(y1_scale)
    ax2.set_yscale(y2_scale)
    # set x ticklabel format
    if x_tickstyle is not None:
        ax1.ticklabel_format(axis='x', style=x_tickstyle, scilimits=(0,0))
    if y1_tickstyle is not None:
        ax1.ticklabel_format(axis='y', style=y1_tickstyle, scilimits=(0,0))
    if y2_tickstyle is not None:
        ax2.ticklabel_format(axis='y', style=y2_tickstyle, scilimits=(0,0))
    if xreverse:
        ax1.invert_xaxis()
    if y1_reverse:
        ax1.invert_yaxis()
    if y2_reverse:
        ax2.invert_yaxis()
    if xlim is not None:
        ax1.set_xlim(*xlim)
    if y1_lim is not None:
        ax1.set_ylim(*y1_lim)
    if y2_lim is not None:
        ax2.set_ylim(*y2_lim)
    # plt.title(title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True)
    fig.tight_layout()

    plt.savefig(filename)
    plt.close()


def plot_stack_bars(
    xs: tuple, ys: dict,
    width: float,
    # colors,
    xlabel=None, ylabel=None,
    filename='stack.png'
    ):

    fig, ax = plt.subplots()
    plt.rc('font', size=16)
    plt.subplots_adjust(0.18, 0.16, 0.96, 0.96)
    
    bottom = np.zeros(len(xs))
    for type, y in ys.items():
        ax.bar(xs, y, width, label=type, bottom=bottom)
        bottom += y

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=20)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=20)

    ax.legend()
    
    fig.tight_layout()


    plt.savefig(filename)
    plt.close()
