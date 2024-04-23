
import numpy as np
import random

from ..network import VertexSet, VertexSource, Task, Network
from ..solver import LinearSolver, PathAugSolver
from ..utils.plot import plot_nx_graph, plot_optimized_network

if __name__ == "__main__":

    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    vsrc = VertexSource.NOEL
    vset = VertexSet(vsrc)
    task = Task(vset, 0.2, (10, 11))
    net = Network(task=task)

    net.cluster_by_nearest(5)
    net.update_edges()
    net.plot(None, None, './result/fig_cluster.png')

    net.segment_edge(150, 150)
    net.update_edges()
    net.update_pairs()
    net.plot(None, None, './result/fig_presolve.png')

    solver_path = PathAugSolver(net, 10, 'length')
    solver_path.solve()
    print(solver_path.obj_val)
    plot_optimized_network(solver_path.network.G, solver_path.m, solver_path.c, filename='./result/fig_solved_path.png')

    solver_flow = LinearSolver(net,)
    solver_flow.solve()
    print(solver_flow.obj_val)
    plot_optimized_network(solver_flow.network.G, solver_flow.m, solver_flow.c, filename='./result/fig_solved_flow.png')
