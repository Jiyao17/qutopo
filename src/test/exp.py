
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

    objs = np.ndarray((len(cluster_nums) + 1, ))
    times = np.ndarray((len(cluster_nums) + 1, ))

    # set all values to nan
    objs.fill(np.nan)
    times.fill(np.nan)

    for i, cluster_num in enumerate(cluster_nums):
        net = copy.deepcopy(network)

        net.connect_nodes_nearest(cluster_num, 1)
        # net.connect_nodes_radius(200, 1)
        net.connect_nearest_component(1)
        net.segment_edges(200, 200, 1)

        start = time.time()
        solver = FlowSolver(net, 600)
        solver.build()
        solver.solve()
        end = time.time()

        objs[cluster_num] = solver.obj_val
        times[cluster_num] = end - start

        print(f"Cluster Number: {cluster_num} Done.")
        
        # plot 2 y axis, obj and time
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(cluster_nums, objs[1:], 'g-')
        ax2.plot(cluster_nums, times[1:], 'b-')
        ax1.set_xlabel('Cluster Number')
        ax1.set_ylabel('Cost', color='g')
        ax2.set_ylabel('Time', color='b')

        prob = network.hw_params['swap_prob']
        plt.savefig(f'./result/flow/fig_{network.task.vset.vsrc.name}_{prob}.png')


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
    demand = 100
    task = Task(vset, 0.5, (demand, demand+1))
    net = Topology(task=task)

    city_num = len(net.graph.nodes)
    seg_len = get_edge_length(demand, net.hw_params['photon_rate'], net.hw_params['fiber_loss'])
    print(f"Suggested edge length: {seg_len}")

    density = 10
    net.connect_nodes_nearest(density) # ~7.8e6
    # net.connect_nearest_component()
    # k=50    k=100   k=150   k=200   k=500
    # ~3.5e7  ~3.2e7  ~2.8e7  ~1e7    ~6.4e6
    # net.make_clique(list(net.graph.nodes(data=False)), 1)
    net.segment_edges(seg_len, seg_len)
    # net.segment_edges(10, 10)
    # net.connect_nodes_radius(200, 1)
    print("node num: ", len(net.graph.nodes))
    print("edge num: ", len(net.graph.edges))

    net.plot(None, None, './result/path/fig.png')

    # print(net.graph.edges(data=True))

    k = 500
    net_path = copy.deepcopy(net)
    start = time.time()
    path_solver = PathSolver(net_path, k, edge_weight='length', mip_gap=0.05, output=True)
    # path_solver = PathSolver(net_path, k, mip_gap=0.01, output=False)
    # solver = PathSolverNonCost(net, k, output=True)
    # solver = PathSolverMinResource(net, k, output=True)
    # path_solver.prepare_paths(swap_func=sequential_swap)

    path_solver.prepare_paths(swap_func=complete_swap)
    path_solver.build()
    path_solver.solve()
    end = time.time()
    path_time = end - start
    print(f"Time elapsed: {end - start}")

    # m = { node: int(m.x) for node, m in path_solver.m.items() }
    # c = { edge: int(c.x) for edge, c in path_solver.c.items() }
    # phi = { edge: phi.x for edge, phi in path_solver.phi.items() }
    # print("Path Solver obj: ", path_solver.obj_val)
    # plot_optimized_network(
    #     path_solver.network.graph,
    #     m, c, phi,
    #     False,
    #     filename='./result/path/fig-solved.png'
    #     )

    net_flow = copy.deepcopy(net)
    start = time.time()
    flow_solver = FlowSolver(net_flow, time_limit=600, mip_gap=0.05, output=True)
    # flow_solver = FlowSolverNonCost(net_flow, mip_gap=0.01)
    # flow_solver = FlowSolverMinResource(net_flow, mip_gap=0.01)
    flow_solver.build()
    flow_solver.solve()
    end = time.time()
    if flow_solver.obj_val is not None:
        print("Flow Solver obj: ", flow_solver.obj_val)
        print("Time elapsed: ", end - start)
        # plot_optimized_network(
        #     flow_solver.network.graph, 
        #     flow_solver.m, flow_solver.c, flow_solver.phi,
        #     filename='./result/test/fig_solved_flow.png')

    # net_greedy = copy.deepcopy(net)
    # start = time.time()
    # # solver = GreedySolver(net, k, None, 'resource')
    # greedy_solver = GreedySolver(net_greedy, k, None, 'cost')
    # greedy_solver.solve()
    # end = time.time()

    
    # print("Greedy Solver obj: ", greedy_solver.obj_val)
    # print("Time elapsed: ", end - start)
    # plot_optimized_network(
    #     greedy_solver.network.graph, 
    #     greedy_solver.m, greedy_solver.c, greedy_solver.phi, 
    #     filename='./result/greedy/fig-solved.png'
    #     )

    # plot objs vs time, obj as y, time as x
    fig = plt.figure()
    ax = fig.add_subplot(111)
    path_times = [path_solver.model._obj_vals[i][0] for i in range(len(path_solver.model._obj_vals))]
    path_objs = [path_solver.model._obj_vals[i][1] for i in range(len(path_solver.model._obj_vals))]
    ax.plot(path_times, path_objs, label='Path Solver')
    # print(path_times)
    # print(path_objs)

    flow_times = [flow_solver.model._obj_vals[i][0] for i in range(len(flow_solver.model._obj_vals))]
    flow_objs =  [flow_solver.model._obj_vals[i][1] for i in range(len(flow_solver.model._obj_vals))]
    ax.plot(flow_times, flow_objs, label='Flow Solver')

    # plot single dot for greedy solver
    # ax.scatter(end - start, greedy_solver.obj_val, label='Greedy Solver')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Objective Value')
    # y scale to log
    ax.set_yscale('log')
    ax.legend()
    plt.savefig('./result/fig_obj_time.png')
    plt.close()



def demo():
    vsrcs = [
    VertexSource.GETNET,
    VertexSource.NOEL, 
    VertexSource.IRIS, 
    VertexSource.ATT, 
    ]

    vsrc = VertexSource.NOEL
    vset = VertexSet(vsrc)
    demand = 10
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
    # path_nums = range(50, 50, 50)
    # test_path_solver_clique(net, path_nums)
    # test_flow_solver(net, cluster_nums)
    # test_greedy_solver(net, cluster_nums, path_nums)
    # 



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

    comp_solvers()

    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)

    # vsrc = VertexSource.NOEL
    # vset = VertexSet(vsrc)
    # demand = 10
    # task = Task(vset, 0.5, (demand, demand+1))
    # net = Topology(task=task)
    
    # test_flow_solver(net, range(1, 11))
