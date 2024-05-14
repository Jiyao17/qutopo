
import numpy as np
import random
import time
import copy

import matplotlib.pyplot as plt

from ..network import *
from ..solver import *

from ..utils.plot import plot_nx_graph, plot_optimized_network


def test_path_solver_clique(
        network: Topology, 
        path_nums: 'list[int]'=range(1, 11),
        seg_len: float=200,
        ):

    objs = {path_num: np.nan for path_num in path_nums}
    times = {path_num: np.nan for path_num in path_nums}

    for j, path_num in enumerate(path_nums):
        net = copy.deepcopy(network)

        # net.connect_nodes_nearest(cluster_num, 1)
        # net.connect_nodes_radius(200, 1)
        # net.connect_nearest_component(1)
        net.make_clique(list(net.graph.nodes), 1)
        net.segment_edges(seg_len, seg_len, 1)

        start = time.time()
        try:
            solver = PathSolver(net, path_num)
            solver.solve()
        except AttributeError:
            print(f"Path Number: {path_num} Failed.")
        else:
            end = time.time()
            times[path_num] = end - start
            objs[path_num] = solver.obj_val

        # plot 2d figure for obj
        fig = plt.figure()
        ax = fig.add_subplot(111)
        objs_arr = np.array([objs[path_num] for path_num in path_nums])
        ax.plot(path_nums, objs_arr)
        ax.set_xlabel('Shortest Path Number')
        ax.set_ylabel('Objective Value')
        plt.savefig(f'./result/path/fig_obj_{network.task.vset.vsrc.name}_clique.png')

        # plot 2d figure for time
        fig = plt.figure()
        ax = fig.add_subplot(111)
        times_arr = np.array([times[path_num] for path_num in path_nums])
        ax.plot(path_nums, times_arr)
        ax.set_xlabel('Shortest Path Number')
        ax.set_ylabel('Time')
        plt.savefig(f'./result/path/fig_time_{network.task.vset.vsrc.name}_clique.png')
        plt.close()
    return objs, times



def test_path_solver_nearest(
        network: Topology, 
        cluster_nums: 'list[int]'=range(1, 11),
        path_nums: 'list[int]'=range(1, 11),
        seg_len: float=200,
        ):

    objs = {(cluster_num, path_num): np.nan for cluster_num in cluster_nums for path_num in path_nums}
    times = {(cluster_num, path_num): np.nan for cluster_num in cluster_nums for path_num in path_nums}

    for i, cluster_num in enumerate(cluster_nums):
        for j, path_num in enumerate(path_nums):

            net = copy.deepcopy(network)

            net.connect_nodes_nearest(cluster_num, 1)
            # net.connect_nodes_radius(150, 1)
            net.connect_nearest_component(1)
            # net.make_clique(list(net.graph.nodes), 1)
            net.segment_edges(seg_len, seg_len, 1)

            start = time.time()
            solver = PathSolver(net, path_num, 'length')
            try:
                solver.solve()
            except AttributeError:
                print(f"Cluster Number: {cluster_num}, Path Number: {path_num} Failed.")
                objs[cluster_num, path_num] = np.nan
                times[cluster_num, path_num] = np.nan
            else:
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
        objs_arr = np.array([[objs[cluster_num, path_num] for path_num in path_nums] for cluster_num in cluster_nums])
        ax.plot_surface(X, Y, objs_arr, cmap='viridis')
        ax.set_xlabel('Cluster Number')
        ax.set_ylabel('Path Number')
        ax.set_zlabel('Price')
        # set z max to 5*min
        # ax.set_zlim(0, 5*np.min(objs[1:, 1:]))
        # set z to log scale
        ax.set_zscale('log')
        # rotate the axes and update
        ax.view_init(30, 70)
        plt.savefig(f'./result/path/fig_obj_{network.task.vset.vsrc.name}.png')


        # plot 3d figure for time
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(cluster_nums, path_nums)
        times_arr = np.array([[times[cluster_num, path_num] for path_num in path_nums] for cluster_num in cluster_nums])
        ax.plot_surface(X, Y, times_arr, cmap='viridis')
        ax.set_xlabel('Cluster Number')
        ax.set_ylabel('Path Number')
        ax.set_zlabel('Time')
        ax.view_init(30, 22.5)
        plt.savefig(f'./result/path/fig_time_{network.task.vset.vsrc.name}.png')

    return objs, times




def test_flow_solver(
        network: Topology,
        cluster_nums: 'list[int]'=range(1, 11),
        ):

    objs = np.zeros((len(cluster_nums) + 1, ))
    times = np.zeros((len(cluster_nums) + 1, ))

    for i, cluster_num in enumerate(cluster_nums):
        net = copy.deepcopy(network)

        net.connect_nodes_nearest(cluster_num, 1)
        # net.connect_nodes_radius(200, 1)
        net.connect_nearest_component(1)
        net.segment_edges(200, 200, 1)

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
    plt.savefig(f'./result/flow/fig_obj_{network.task.vset.vsrc.name}.png')

    # plot line figure for time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cluster_nums, times[1:])
    ax.set_xlabel('Cluster Number')
    ax.set_ylabel('Time')
    plt.savefig(f'./result/flow/fig_time_{network.task.vset.vsrc.name}.png')

    return objs, times


def test_greedy_solver(
        network: Topology, 
        cluster_nums: 'list[int]'=range(1, 11),
        path_nums: 'list[int]'=range(1, 11),
        ):

    times = np.zeros((len(cluster_nums) + 1, len(path_nums) + 1))
    objs = np.zeros((len(cluster_nums) + 1, len(path_nums) + 1))

    for i, cluster_num in enumerate(cluster_nums):
        for j, path_num in enumerate(path_nums):
            net = copy.deepcopy(network)

            net.connect_nodes_nearest(cluster_num, 1)
            net.connect_nodes_radius(200, 1)
            net.connect_nearest_component(1)
            net.segment_edges(200, 200, 1)

            start = time.time()
            solver = GreedySolver(net, path_num)
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
    # set z max to 5*min
    ax.set_zlim(0, 5*np.min(objs[1:, 1:]))
    # rotate the axes and update
    ax.view_init(30, 70)
    plt.savefig(f'./result/greedy/fig_obj_{network.task.vset.vsrc.name}.png')


    # plot 3d figure for time
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(cluster_nums, path_nums)
    ax.plot_surface(X, Y, times[1:, 1:].T, cmap='viridis')
    ax.set_xlabel('Cluster Number')
    ax.set_ylabel('Shortest Path Number')
    ax.set_zlabel('Time')
    ax.view_init(30, 45)
    plt.savefig(f'./result/greedy/fig_time_{network.task.vset.vsrc.name}.png')

    return objs, times


def comp_solvers():

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    vsrc = VertexSource.NOEL
    vset = VertexSet(vsrc)
    task = Task(vset, 0.2, (100, 101))
    net = Topology(task=task)

    net.connect_nodes_nearest(5)
    net.connect_nearest_component()
    net.segment_edges(150, 150)

    net_path = copy.deepcopy(net)
    path_solver = PathSolver(net_path, 10, mip_gap=0.01)
    # path_solver = PathSolverNonCost(net_path, 10, mip_gap=0.01)
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
    
    vsrc = VertexSource.NOEL
    vset = VertexSet(vsrc)
    demand = 100
    task = Task(vset, 1.0, (demand, demand+1))
    net = Topology(task=task)

    seg_len = get_edge_length(demand, 1e4, 0.2)
    
    # cluster_nums = range(1, 6)
    # path_nums = range(1, 6)
    cluster_nums = range(5, 16)
    path_nums = range(5, 51)
    
    # for vsrc in vsrcs:
    #     test_path_solver(vsrc)

    # for vsrc in vsrcs:
    #     test_flow_solver(vsrc)

    # test_path_solver_nearest(net, cluster_nums, path_nums, seg_len)
    path_nums = range(50, 1050, 50)
    test_path_solver_clique(net, path_nums)
    # test_flow_solver(net, cluster_nums)
    # test_greedy_solver(net, cluster_nums, path_nums)
    # 
    # comp_solvers()
    print("Done.")
