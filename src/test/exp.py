
import numpy as np
import random
import time
import copy

import matplotlib.pyplot as plt

from ..network import *
from ..solver import *

from ..utils.plot import plot_nx_graph, plot_optimized_network


def test_path_solver(
        network: Network, 
        cluster_nums: 'list[int]'=[1, 2, 3, 4, 5],
        path_nums: 'list[int]'=[1, 2, 3, 4, 5],
        ):

    times = np.zeros((len(cluster_nums) + 1, len(path_nums) + 1))
    objs = np.zeros((len(cluster_nums) + 1, len(path_nums) + 1))


    for i, cluster_num in enumerate(cluster_nums):
        for j, path_num in enumerate(path_nums):
            net = copy.deepcopy(network)

            net.cluster_by_nearest(cluster_num)
            net.nearest_components()
            net.segment_edge(150, 150)

            start = time.time()
            solver = PathSolver(net, path_num)
            solver.solve()
            end = time.time()

            objs[cluster_num, path_num] = solver.obj_val
            times[cluster_num, path_num] = end - start

        print(f"Cluster Number: {cluster_num} Done.")
    # objs[5, :] = 1
    # plot 3d figure for obj
    # obj as z, cluster as x, path as y
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(cluster_nums, path_nums)
    ax.plot_surface(X, Y, objs[1:, 1:].T, cmap='viridis')
    ax.set_xlabel('Cluster Number')
    ax.set_ylabel('Shortest Path Number')
    ax.set_zlabel('Objective Value')
    # rotate the axes and update
    ax.view_init(30, 45)
    plt.savefig(f'./result/path/fig_obj_{vsrc.name}.png')


    # plot 3d figure for time
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(cluster_nums, path_nums)
    ax.plot_surface(X, Y, times[1:, 1:].T, cmap='viridis')
    ax.set_xlabel('Cluster Number')
    ax.set_ylabel('Shortest Path Number')
    ax.set_zlabel('Time')
    ax.view_init(30, 45)
    plt.savefig(f'./result/path/fig_time_{vsrc.name}.png')

    return objs, times


def test_flow_solver(
        vsrc: VertexSource,
        cluster_nums: 'list[int]'=[1, 2, 3, 4, 5],
        ):
    
    vset = VertexSet(vsrc)
    task = Task(vset, 1, (100, 101))
    net = Network(task=task)

    objs = np.zeros((len(cluster_nums) + 1, ))
    times = np.zeros((len(cluster_nums) + 1, ))

    for i, cluster_num in enumerate(cluster_nums):
        net = copy.deepcopy(net)

        net.cluster_by_nearest(cluster_num)
        net.nearest_components()
        net.segment_edge(150, 150)

        start = time.time()
        solver = FlowSolver(net)
        solver.solve()
        end = time.time()

        objs[cluster_num] = solver.obj_val
        times[cluster_num] = end - start

        print(f"Cluster Number: {cluster_num} Done.")
        
    # plot line figure for obj
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cluster_nums, objs[1:])
    ax.set_xlabel('Cluster Number')
    ax.set_ylabel('Objective Value')
    plt.savefig(f'./result/flow/fig_obj_{vsrc.name}.png')

    # plot line figure for time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cluster_nums, times[1:])
    ax.set_xlabel('Cluster Number')
    ax.set_ylabel('Time')
    plt.savefig(f'./result/flow/fig_time_{vsrc.name}.png')

    return objs, times



def comp_solvers():

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    vsrc = VertexSource.NOEL
    vset = VertexSet(vsrc)
    task = Task(vset, 0.2, (100, 101))
    net = Network(task=task)

    net.cluster_by_nearest(5)
    net.nearest_components()
    net.segment_edge(150, 150)

    net_path = copy.deepcopy(net)
    # path_solver = PathSolver(net_path, 10, mip_gap=0.01)
    path_solver = PathSolverNonCost(net_path, 10, mip_gap=0.01)
    # path_solver = PathSolverMinResource(net_path, 10, mip_gap=0.01)
    path_solver.solve()
    print(path_solver.obj_val)
    plot_optimized_network(
        path_solver.network.G, 
        path_solver.m, path_solver.c, path_solver.phi,
        filename='./result/test/fig_solved_path.png')

    net_flow = copy.deepcopy(net)
    # flow_solver = FlowSolver(net_flow, mip_gap=0.01)
    flow_solver = FlowSolverNonCost(net_flow, mip_gap=0.01)
    # flow_solver = FlowSolverMinResource(net_flow, mip_gap=0.01)

    flow_solver.solve()

    if flow_solver.obj_val is not None:
        print(flow_solver.obj_val)
        plot_optimized_network(
            flow_solver.network.G, 
            flow_solver.m, flow_solver.c, flow_solver.phi,
            filename='./result/test/fig_solved_flow.png')


if __name__ == "__main__":
    # weekly conclusion

    # for path solver
    # 1. increasing path number does not reduce the objective value much
    # 2. but increasing cluster number does
    # 3. denser graph is better

    # for flow solver
    # 1. if optimally solved, obj is theoretically better than the one given by path solver
    # 2. solving is fast but building is slow
    #    building time is sensitive to graph density


    vsrcs = [
        VertexSource.GETNET,
        VertexSource.NOEL, 
        VertexSource.IRIS, 
        VertexSource.ATT, 
        ]
    
    # for vsrc in vsrcs:
    #     test_path_solver(vsrc)

    # for vsrc in vsrcs:
    #     test_flow_solver(vsrc)

    # test_path_solver(VertexSource.NOEL)
    
    comp_solvers()
    print("Done.")
