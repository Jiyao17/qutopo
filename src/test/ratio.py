
import time
import random
import copy

import multiprocessing as mp

import numpy as np

from ..network import Topology, VertexSource, VertexSet, Task
from ..solver import PathSolver, FlowSolver, GreedySolver

from ..network import get_edge_length
from ..network.quantum import complete_swap, sequential_swap, HWParam

from ..utils.plot import plot_stack_bars



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

    node_budget = solver.node_budget_val
    edge_budget = solver.edge_budget_val
    
    return total_time, times, objs, node_budget, edge_budget

def compare_price_ratio(params, repeat=1):
    """
    This set of evaluations compares different price ratios between memory and channel
        -1:10
        -1:1
        -10:1
    """
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)

    
    density = 2
    k = 100
    ratios = [1/10, 1, 10]

    node_budgets = np.zeros((len(ratios), repeat)) * np.nan
    edge_budgets = np.zeros((len(ratios), repeat)) * np.nan

    for i, ratio in enumerate(ratios):
        print(f"ratio {ratio}")
        
        results = []
        pool = mp.Pool(repeat)

        for j in range(repeat):
            print(f"repeat {j}")

            vsrc = VertexSource.NOEL
            vset = VertexSet(vsrc)
            demand = 10
            task = Task(vset, 0.5, (demand, demand+1))
            params['pm'] = ratio
            params['pm_install'] = ratio*10
            params['pc'] = 1
            params['pc_install'] = 10
            net = Topology(task=task, hw_params=params)
            # seg_len = get_edge_length(demand, net.hw_params['photon_rate'], net.hw_params['fiber_loss'])
            seg_len = 150
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
            total_time, times, objs, node_budget, edge_budget = result.get()
            if node_budget is not None and edge_budget is not None:
                node_budgets[i, j] = node_budget
                edge_budgets[i, j] = edge_budget

        # average the results
        avg_node_budgets = np.nanmean(node_budgets, axis=1)
        avg_edge_budgets = np.nanmean(edge_budgets, axis=1)

        # prepare for plotting
        xs = ('1:10', '1:1', '10:1')
        ys = {'Node Budget': avg_node_budgets, 'Edge Budget': avg_edge_budgets}

        # plot
        plot_stack_bars(
            xs, ys, 
            0.6, 
            # colors=['blue', 'green'],
            xlabel='Price Ratio', ylabel='Budget',
            filename=f'ratio.png'
            )
        # percentage
        avg_total_budgets = avg_node_budgets + avg_edge_budgets
        ys = {'Node Budget': avg_node_budgets/avg_total_budgets, 'Edge Budget': avg_edge_budgets/avg_total_budgets}
        plot_stack_bars(
            xs, ys, 
            0.6, 
            # colors=['blue', 'green'],
            xlabel='Price Ratio', ylabel='Budget Percentage',
            filename=f'ratio_percentage.png'
            )
if __name__ == '__main__':
    # set gurobi environment
    # mute parameter setting
    import gurobipy as gp
    gp.setParam('OutputFlag', 0)

    params = copy.deepcopy(HWParam)
    params['swap_prob'] = 0.75
    compare_price_ratio(params, repeat=1)
