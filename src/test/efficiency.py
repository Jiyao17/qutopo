
import time
import random
import copy
import pickle

import multiprocessing as mp

import numpy as np
import gurobipy as gp


from ..network import Topology, VertexSource, VertexSet, Task, VertexSetRandom
from ..solver import PathSolver, FlowSolver, GreedySolver

from ..network import get_edge_length
from ..network.quantum import complete_swap, sequential_swap, HWParam

from ..utils.plot import plot_2y_lines, plot_lines



# evaluations

def run_flow_solver(
    network: Topology,
    time_limit=1000,
    mip_gap=0.01,
    ):
    start = time.time()

    solver = FlowSolver(network, time_limit=time_limit, mip_gap=mip_gap)
    solver.build()
    solver.solve()

    end = time.time()

    total_time = end - start
    times = [ solver.obj_vals[i][0] for i in range(len(solver.obj_vals)) ]
    objs = [ solver.obj_vals[i][1] for i in range(len(solver.obj_vals)) ]

    return total_time, times, objs

def run_path_solver(
    network: Topology,
    k = 100, 
    edge_weight='length', 
    time_limit=1000,
    mip_gap=0.01,
    swap_func=complete_swap,
    ):

    start = time.time()
    solver = PathSolver(network, k=k, edge_weight=edge_weight, time_limit=time_limit, mip_gap=mip_gap)

    solver.prepare_paths(swap_func=swap_func)
    solver.build()
    solver.solve()
    end = time.time()

    total_time = end - start
    times = [ solver.obj_vals[i][0] for i in range(len(solver.obj_vals)) ]
    objs = [ solver.obj_vals[i][1] for i in range(len(solver.obj_vals)) ]

    return total_time, times, objs

def run_greedy_solver(
    network: Topology,
    k = 100, 
    edge_weight='length', 
    swap_func=complete_swap,
    ):

    start = time.time()

    greedy_solver = GreedySolver(network, k, edge_weight, greedy_opt='cost', swap_func=swap_func)
    greedy_solver.solve()

    end = time.time()

    total_time = end - start
    obj = greedy_solver.obj_val

    return total_time, obj

def compare_efficiency(node_num, params, repeat=1):
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

    # densities = [2, 4, 6, 8, 10]
    densities = range(2, 11, 2)
    # densities = [1, 2, 3, 4, 5, 6]
    flow_objs = np.zeros((len(densities), repeat)) * np.nan
    flow_times = np.zeros((len(densities), repeat)) * np.nan
    opt_path_objs = np.zeros((len(densities), repeat)) * np.nan
    opt_path_times = np.zeros((len(densities), repeat)) * np.nan
    seq_path_objs = np.zeros((len(densities), repeat)) * np.nan
    seq_path_times = np.zeros((len(densities), repeat)) * np.nan
    opt_greedy_objs = np.zeros((len(densities), repeat)) * np.nan
    opt_greedy_times = np.zeros((len(densities), repeat)) * np.nan
    seq_greedy_objs = np.zeros((len(densities), repeat)) * np.nan
    seq_greedy_times = np.zeros((len(densities), repeat)) * np.nan

    # vsrc = VertexSource.NOEL
    # vset = VertexSet(vsrc)
    vset = VertexSetRandom(node_num)
    vset.scale((0.01, 0.01))
    demand = 10
    task = Task(vset, 0.5, (demand, demand+1))
    raw_net = Topology(task=task, hw_params=params)

    flow_density_control = 2
    if node_num <= 50:
        flow_density_control = 4
    if node_num <= 35:
        flow_density_control = 6
    if node_num <= 30:
        flow_density_control = 8
    if node_num <= 20:
        flow_density_control = 10
    if node_num <= 10:
        flow_density_control = 10

    for i, density in enumerate(densities):
        print(f"density {density}")

        for j in range(repeat):
            print(f"repeat {j}")
            pool = mp.Pool(5)

            net = copy.deepcopy(raw_net)
            # seg_len = get_edge_length(demand, net.hw_params['photon_rate'], net.hw_params['fiber_loss'])
            seg_len = 150
            net.connect_nodes_nearest(density)
            # make sure the network is connected
            net.connect_nearest_component()
            net.segment_edges(seg_len, seg_len)

            # run each solver in one process
            if density <= flow_density_control:
                flow_path_result = pool.apply_async(run_flow_solver, args=(net, 600))
            k = 100
            opt_path_result = pool.apply_async(run_path_solver, args=(net, k))
            seq_path_result = pool.apply_async(
                run_path_solver, 
                args=(net, k),
                kwds={'swap_func': sequential_swap}
                )
            opt_greedy_result = pool.apply_async(run_greedy_solver, args=(net, k))
            seq_greedy_result = pool.apply_async(
                run_greedy_solver, 
                args=(net, k),
                kwds={'swap_func': sequential_swap}
                )

            # wait for all processes to finish
            pool.close()
            pool.join()

            # fetch all results
            if density <= flow_density_control:
                total_time, times, objs = flow_path_result.get()
                if len(objs) > 0:
                    flow_objs[i, j] = objs[-1]
                    flow_times[i, j] = total_time
            
            total_time, times, objs = opt_path_result.get()
            if len(objs) > 0:
                opt_path_objs[i, j] = objs[-1]
                opt_path_times[i, j] = total_time
            
            total_time, times, objs = seq_path_result.get()
            if len(objs) > 0:
                seq_path_objs[i, j] = objs[-1]
                seq_path_times[i, j] = total_time

            total_time, obj = opt_greedy_result.get()
            if len(objs) > 0:
                opt_greedy_objs[i, j] = obj
                opt_greedy_times[i, j] = total_time

            total_time, obj = seq_greedy_result.get()
            if obj is not None:
                seq_greedy_objs[i, j] = obj
                seq_greedy_times[i, j] = total_time


            # print(opt_path_objs)
            # print(opt_path_times)
            # take average
            avg_flow_objs = np.nanmean(flow_objs, axis=1)
            avg_flow_times = np.nanmean(flow_times, axis=1)
            avg_opt_path_objs = np.nanmean(opt_path_objs, axis=1)
            avg_opt_path_times = np.nanmean(opt_path_times, axis=1)
            avg_seq_path_objs = np.nanmean(seq_path_objs, axis=1)
            avg_seq_path_times = np.nanmean(seq_path_times, axis=1)
            avg_opt_greedy_objs = np.nanmean(opt_greedy_objs, axis=1)
            avg_opt_greedy_times = np.nanmean(opt_greedy_times, axis=1)
            avg_seq_greedy_objs = np.nanmean(seq_greedy_objs, axis=1)
            avg_seq_greedy_times = np.nanmean(seq_greedy_times, axis=1)

            
            # print(avg_opt_path_objs)
            # print(avg_opt_path_times)

            # prepare for plotting
            ys1 = [avg_flow_objs, avg_seq_path_objs, avg_opt_path_objs, avg_seq_greedy_objs, avg_opt_greedy_objs]
            ys2 = [avg_flow_times, avg_seq_path_times, avg_opt_path_times, avg_seq_greedy_times, avg_opt_greedy_times]
            y1_labels = ['FLOW', 'PATH-SEQ', 'PATH-OPT', 'GREEDY-SEQ', 'GREEDY-OPT']
            y2_labels = None

            pickle.dump(
                (densities, ys1, ys2, y1_labels, y2_labels),
                open(f'./result/efficiency/efficiency-{vset.name}.pkl', 'wb')
            )

            plot_lines(
                densities, ys1,
                'Density', 'Objective', 
                y1_labels,
                yticklabel='sci',
                adjust=(0.16, 0.16, 0.96, 0.92),
                filename=f'./result/efficiency/efficiency-objective-{vset.name}.png'
            )

            plot_lines(
                densities, ys2,
                'Density', 'Time', 
                y1_labels,
                yscale='log',
                adjust=(0.18, 0.16, 0.94, 0.94),
                filename=f'./result/efficiency/efficiency-time-{vset.name}.png'
            )
                

            plot_2y_lines(
                densities, ys1, ys2,
                'Density', 'Objective', 'Time',  
                y1_labels, y2_labels,
                ['-', '-', '-', '-', '-'], ['--', '--', '--', '--', '--'],
                ['blue', 'green', 'red', 'orange', 'purple'], ['blue', 'green', 'red', 'orange', 'purple'],
                ['o', 's', '^', 'v', 'x'], ['o', 's', '^', 'v', 'x'],
                xscale='linear', y1_scale='linear', y2_scale='log',
                xreverse=False, y1_reverse=False, y2_reverse=False,
                xlim=None, y1_lim=None, y2_lim=None,
                filename='efficiency.png'
            )


def load_plot_result(node_num):
    densities, ys1, ys2, y1_labels, y2_labels = pickle.load(
        open(f'./result/efficiency/efficiency-random{node_num}.pkl', 'rb')
        )
    
    plot_lines(
        densities, ys1,
        'Density', 'Objective', 
        y1_labels,
        yticklabel='sci',
        adjust=(0.18, 0.16, 0.96, 0.92),
        filename=f'./result/efficiency/efficiency-objective-random{node_num}.png'
    )

    plot_lines(
        densities, ys2,
        'Density', 'Time', 
        y1_labels,
        yscale='log',
        adjust=(0.18, 0.16, 0.94, 0.94),
        filename=f'./result/efficiency/efficiency-time-random{node_num}.png'
    )


if __name__ == '__main__':
    # use time as random seed
    random.seed(int(time.time()))
    np.random.seed(int(time.time()))

    # set gurobi environment
    # mute parameter setting
    gp.setParam('OutputFlag', 0)

    params = copy.deepcopy(HWParam)
    params['swap_prob'] = 0.75

    # node_nums = [20, 35, 50]
    node_nums = [10, 20, 30, 35, 50]
    for node_num in node_nums:
        compare_efficiency(node_num, params, repeat=1)
        # load_plot_result(node_num)
