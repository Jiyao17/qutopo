
import time
import random
import copy

import multiprocessing as mp

import numpy as np

from ..network import Topology, VertexSource, VertexSet, Task, VertexSetRandom
from ..solver import PathSolver, FlowSolver, GreedySolver

from ..network import get_edge_length
from ..network.quantum import complete_swap, sequential_swap, HWParam

from ..utils.plot import plot_2y_lines



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
    
    return total_time, times, objs

def compare_segmentation_lengths(params, repeat=1):
    """
    This set of evaluations compares different network scales
    """
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)

    
    density = 5
    k = 100
    scales = [ 0.005, 0.01, 0.015, 0.02, ]
    seg_len = 150

    opt_path_objs = np.zeros((len(scales), repeat)) * np.nan
    opt_path_times = np.zeros((len(scales), repeat)) * np.nan

    for i, scale in enumerate(scales):
        print(f"seg_len {scale}")
    
        vset = VertexSetRandom(num=20,)
        vset.scale((scale, scale))
        
        pool = mp.Pool(repeat)
        results = []

        for j in range(repeat):
            print(f"repeat {j}")

            demand = 10
            task = Task(vset, 0.5, (demand, demand+1))
            net = Topology(task=task, hw_params=params)
            # seg_len = get_edge_length(demand, net.hw_params['photon_rate'], net.hw_params['fiber_loss'])

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
            total_time, times, objs = result.get()
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
            scales, ys1, ys2,
            'Scale', 'Objective', 'Time', 
            y1_labels, y2_labels,
            y1_stypes, y2_stypes,
            y1_colors, y2_colors,
            y1_markers, y2_markers,
            xscale='linear', y1_scale='linear', y2_scale='linear',
            xreverse=False, y1_reverse=False, y2_reverse=False,
            xlim=None, y1_lim=None, y2_lim=None,
            filename='scale.png'
        )

if __name__ == '__main__':
    # set gurobi environment
    # mute parameter setting
    import gurobipy as gp
    gp.setParam('OutputFlag', 0)

    params = copy.deepcopy(HWParam)
    params['swap_prob'] = 0.75
    compare_segmentation_lengths(params, repeat=3)
