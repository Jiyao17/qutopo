

import networkx as nx
import gurobipy as gp
import matplotlib.pyplot as plt

from ..network import *
from ..optimizer.task import QuTopoTask
from ..optimizer.solver import LinearSolver

from ..utils.plot import plot_opt_result

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
    grd = GroundNetTopo(GroundNetOpt.NOEL)
    gps = ConstellationPosition(ConstellationOpt.GPS)
    topo = FusedTopo(grd, gps)
    task = QuTopoTask(topo, 1)

    plot_opt_result(topo, None, filename='./result/fig.png')

    solver = LinearSolver(task)
    solver.optimize(obj='flow')
    print(solver.model.objVal)
    plot_opt_result(topo, solver, filename='./result/fig.png')

if __name__ == "__main__":
    # test_topo()
    test_solver()
    # import numpy as np
    # rate =  10 * 1000000 * np.power(10, - 0.2 * 600 / 10 ) ** 2
    # print(rate)
