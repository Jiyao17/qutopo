
import time
import random
import copy

import multiprocessing as mp

import numpy as np

from ..network import Topology, VertexSource, VertexSet, Task
from ..solver import PathSolver, FlowSolver, GreedySolver

from ..network import get_edge_length
from ..network.quantum import complete_swap, sequential_swap, HWParam

from ..utils.plot import plot_2y_lines



# evaluations

def run_flow_solver(
    pid: int,
    queue: mp.Queue,
    network: Topology,
    time_limit=600,
    mip_gap=0.05,
    ):
    start = time.time()

    solver = FlowSolver(network, time_limit=time_limit, mip_gap=mip_gap)
    solver.build()
    solver.solve()

    end = time.time()

    total_time = end - start
    times = [ solver.model._obj_vals[i][0] for i in range(len(solver.model._obj_vals)) ]
    objs = [ solver.model._obj_vals[i][1] for i in range(len(solver.model._obj_vals)) ]

    queue.put((pid, total_time, times, objs))

def run_path_solver(
    pid: int,
    queue: mp.Queue,
    network: Topology,
    k = 100, 
    edge_weight='length', 
    time_limit=600,
    mip_gap=0.05,
    swap_func=complete_swap,
    ):

    start = time.time()
    solver = PathSolver(network, k=k, edge_weight=edge_weight, time_limit=time_limit, mip_gap=mip_gap)

    solver.prepare_paths(swap_func=swap_func)
    solver.build()
    solver.solve()
    end = time.time()

    total_time = end - start
    times = [ solver.model._obj_vals[i][0] for i in range(len(solver.model._obj_vals)) ]
    objs = [ solver.model._obj_vals[i][1] for i in range(len(solver.model._obj_vals)) ]

    queue.put((pid, total_time, times, objs))

def run_greedy_solver(network, config):
    pass

def compare_efficiency(params, repeat=1):
    """
    This set of evaluations compares the efficiency of
        -flow solver
        -path solver
        -greedy solver
    on the given network and configuration.
    """
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)


    vsrc = VertexSource.NOEL
    vset = VertexSet(vsrc)
    demand = 10
    task = Task(vset, 0.5, (demand, demand+1))
    net = Topology(task=task, hw_params=params)

    seg_len = get_edge_length(demand, net.hw_params['photon_rate'], net.hw_params['fiber_loss'])

    # densities = [2, 4, 6, 8, 10]
    # densities = [2, 4, 6, 8]
    densities = [1, 2, 3, 4, 5, 6]
    flow_objs = np.zeros((repeat, len(densities))) * np.nan
    flow_times = np.zeros((repeat, len(densities))) * np.nan
    opt_path_objs = np.zeros((repeat, len(densities))) * np.nan
    opt_path_times = np.zeros((repeat, len(densities))) * np.nan
    seq_path_objs = np.zeros((repeat, len(densities))) * np.nan
    seq_path_times = np.zeros((repeat, len(densities))) * np.nan
    greedy_objs = np.zeros((repeat, len(densities))) * np.nan
    greedy_times = np.zeros((repeat, len(densities))) * np.nan

    for i in range(repeat):
        print(f"repeat {i}")
        for j, density in enumerate(densities):
            print(f"density {density}")

            queue = mp.Queue(3)
            pool = mp.Pool(3)

            net_temp = copy.deepcopy(net)
            net_temp.connect_nodes_nearest(density)
            # make sure the network is connected
            net_temp.connect_nearest_component()
            net_temp.segment_edges(seg_len, seg_len)
            # print("node num: ", len(net.graph.nodes))
            # print("edge num: ", len(net.graph.edges))

            # run each solver in one process
            # p = mp.Process(target=run_flow_solver, args=(0, queue, net_temp))
            pool.apply_async(run_flow_solver, args=(0, queue, net_temp))

            # p = mp.Process(
            #     target=run_path_solver, 
            #     args=(1, queue, net_temp)
            #     )
            pool.apply_async(run_path_solver, args=(1, queue, net_temp))

            # p = mp.Process(
            #     target=run_path_solver, 
            #     args=(2, queue, net_temp), 
            #     kwargs={'swap_func': sequential_swap}
            #     )
            pool.apply_async(
                run_path_solver, 
                args=(2, queue, net_temp), 
                kwds={'swap_func': sequential_swap}
                )

            # p = mp.Process(target=run_greedy_solver, args=(3, queue, net_temp))
            # p.start()

            # wait for all processes to finish
            pool.close()
            pool.join()

            # fetch all results
            for _ in range(queue.qsize()):
                pid, total_time, times, objs = queue.get()
                if pid == 0:
                    flow_objs[i, j] = objs[-1]
                    flow_times[i, j] = total_time
                elif pid == 1:
                    opt_path_objs[i, j] = objs[-1]
                    opt_path_times[i, j] = total_time
                elif pid == 2:
                    seq_path_objs[i, j] = objs[-1]
                    seq_path_times[i, j] = total_time
                # elif pid == 3:
                #     greedy_objs[i, j] = objs[-1]
                #     greedy_times[i, j] = total_time

            # total_time, times, objs = run_path_solver(net_temp)
            # opt_path_objs[i, j] = objs[-1]
            # opt_path_times[i, j] = total_time

            # total_time, times, objs = run_path_solver(net_temp, swap_func=sequential_swap)
            # seq_path_objs[i, j] = objs[-1]
            # seq_path_times[i, j] = total_time

            # total_time, times, objs = run_flow_solver(net_temp)
            # flow_objs[i, j] = objs[-1]
            # flow_times[i, j] = total_time
    
    # print(path_objs)
    # print(path_times)
    # take average
    flow_objs = np.nanmean(flow_objs, axis=0)
    flow_times = np.nanmean(flow_times, axis=0)
    opt_path_objs = np.nanmean(opt_path_objs, axis=0)
    opt_path_times = np.nanmean(opt_path_times, axis=0)
    seq_path_objs = np.nanmean(seq_path_objs, axis=0)
    seq_path_times = np.nanmean(seq_path_times, axis=0)
    greedy_objs = np.nanmean(greedy_objs, axis=0)
    greedy_times = np.nanmean(greedy_times, axis=0)
    # print(path_objs)
    # print(path_times)

    # prepare for plotting
    ys1 = [flow_objs, opt_path_objs, seq_path_objs, greedy_objs]
    ys2 = [flow_times, opt_path_times, seq_path_times, greedy_times]
    y1_labels = ['FLOW', 'PATH-OPT', 'PATH-SEQ', 'GREEDY']
    y2_labels = None

    plot_2y_lines(
        densities, ys1, ys2,
        'Density', 'Objective', 'Time',  
        y1_labels, y2_labels,
        ['-', '-', '-', '-'], ['--', '--', '--', '--'],
        ['blue', 'green', 'red', 'orange'], ['blue', 'green', 'red', 'orange'],
        ['o', 's', '^', 'v'], ['o', 's', '^', 'v'],
        xscale='linear', y1_scale='linear', y2_scale='log',
        xreverse=False, y1_reverse=False, y2_reverse=False,
        xlim=None, y1_lim=None, y2_lim=None,
        filename='efficiency.png'
    )


if __name__ == '__main__':
    # set gurobi environment
    # mute parameter setting
    import gurobipy as gp
    gp.setParam('OutputFlag', 0)

    params = copy.deepcopy(HWParam)
    params['swap_prob'] = 0.75
    compare_efficiency(params=params, repeat=1)