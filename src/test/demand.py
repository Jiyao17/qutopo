
import time
import random
import copy

import multiprocessing as mp

import numpy as np

from ..network import Topology, VertexSource, VertexSet, Task, VertexSetRandom
from ..solver import PathSolver, FlowSolver, GreedySolver

from ..network import get_edge_length
from ..network.quantum import complete_swap, sequential_swap, HWParam

from ..utils.plot import plot_2y_lines, plot_simple_topology



# evaluations


def run_path_solver(
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
    times = [ solver.obj_vals[i][0] for i in range(len(solver.obj_vals)) ]
    objs = [ solver.obj_vals[i][1] for i in range(len(solver.obj_vals)) ]

    Im = {node: 1 if solver.m[node].x > 0 else 0 for node in solver.m}
    Ic = {edge: 1 if solver.c[edge].x > 0 else 0 for edge in solver.c}

    
    return total_time, times, objs, Im, Ic

def compare_demand_intensity(params, repeat=1):
    """
    This set of evaluations compares different network scales
    """
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    
    density = 5
    k = 100
    fracs = [0.1, 0.5, 1]
    names = ['LOW', 'MEDIUM', 'HIGH']
    demands = [ 1, 10, 100,]
    seg_len = 150

    opt_path_objs = np.zeros((len(demands), repeat)) * np.nan
    opt_path_times = np.zeros((len(demands), repeat)) * np.nan

    for i, demand, frac in zip(range(len(demands)), demands, fracs):
        print(f"demand {demand}")
    
        pool = mp.Pool(repeat)
        results = []

        for j in range(repeat):
            print(f"repeat {j}")

            vsrc = VertexSource.NOEL
            vset = VertexSet(vsrc)
            task = Task(vset, frac, (demand, demand+1))
            net = Topology(task=task, hw_params=params)

            net.connect_nodes_nearest(density)
            # make sure the network is connected
            net.connect_nearest_component()
            net.segment_edges(seg_len, seg_len)

            result = pool.apply_async(run_path_solver, args=(net, k))
            results.append(result)

        # wait for all processes to finish
        pool.close()
        pool.join()

        # fetch all results
        for j, result in enumerate(results):
            total_time, times, objs, Im, Ic = result.get()
            if len(objs) > 0:
                opt_path_objs[i, j] = objs[-1]
                opt_path_times[i, j] = total_time

        # average the results
        avg_opt_path_objs = np.nanmean(opt_path_objs, axis=1)
        avg_opt_path_times = np.nanmean(opt_path_times, axis=1)

        # prepare for plotting
        ys1 = [avg_opt_path_objs,]
        ys2 = [avg_opt_path_times,]
        y1_labels = ['Objective',]
        y2_labels = ['Time',]

        y1_stypes = ['-',] * len(y1_labels)
        y2_stypes = ['--',] * len(y2_labels)
        y1_colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
        y2_colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
        y1_markers = ['o', 's', '^', 'v', 'x', '+']
        y2_markers = ['o', 's', '^', 'v', 'x', '+']
        plot_2y_lines(
            demands, ys1, ys2,
            'Demands', 'Objective', 'Time', 
            y1_labels, y2_labels,
            y1_stypes, y2_stypes,
            y1_colors, y2_colors,
            y1_markers, y2_markers,
            xscale='log', y1_scale='log', y2_scale='log',
            xreverse=False, y1_reverse=False, y2_reverse=False,
            xlim=None, y1_lim=None, y2_lim=None,
            filename='demand.png'
        )

        users = set()
        for pair, demand in task.D.items():
            if demand > 0:
                users.add(pair[0])
                users.add(pair[1])
        plot_simple_topology(net.graph, Im, Ic, users, filename=f'./result/demand/topology_{names[i]}.png')

if __name__ == '__main__':
    # set gurobi environment
    # mute parameter setting
    import gurobipy as gp
    gp.setParam('OutputFlag', 0)

    params = copy.deepcopy(HWParam)
    compare_demand_intensity(params, repeat=1)
