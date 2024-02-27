

import networkx as nx
import gurobipy as gp
import matplotlib.pyplot as plt

from ..network import *
from ..optimizer.task import QuTopoTask
from ..optimizer.solver import LinearSolver

def test_topo():
    att = GroundNetTopo(GroundNetOpt.ATT)
    gps = ConstellationPosition(ConstellationOpt.GPS)
    net = FusedTopo(att, gps)
    node0 = net.net.nodes['0']
    node1 = net.net.nodes['1']
    print(node0)
    print(node1)
    # get all edges between node0 and node1
    edges = net.net.edges(data=True, keys=True)
    print(edges)


def test_solver():
    grd = GroundNetTopo(GroundNetOpt.GETNET)
    gps = ConstellationPosition(ConstellationOpt.GPS)
    topo = FusedTopo(grd, gps)
    task = QuTopoTask(topo, 1)
    solver = LinearSolver(task)
    solver.export_task()
    solver.optimize()
    print(solver.model.objVal)

    # set edge capacity to optimization result
    for edge in topo.net.edges:
        topo.net.edges[edge]['capacity'] = solver.c[edge].x
    # set node capacity to optimization result
    for node in topo.net.nodes:
        topo.net.nodes[node]['capacity'] = solver.m[node].x
    # draw the multigraph, save the result to a file
    # set capacity as the edge and node labels
    pos = nx.spring_layout(topo.net)
    edge_labels = {(edge[0], edge[1]): int(topo.net.edges[edge]['capacity']) for edge in topo.net.edges}
    node_labels = {node: int(topo.net.nodes[node]['capacity']) for node in topo.net.nodes}
    nx.draw(topo.net, pos, with_labels=False, node_color='lightblue', node_size=500, edge_color='grey', width=1, edge_cmap=plt.cm.Blues)
    nx.draw_networkx_edge_labels(topo.net, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(topo.net, pos, labels=node_labels)


    plt.savefig('test.png')



if __name__ == "__main__":
    # test_topo()
    test_solver()
