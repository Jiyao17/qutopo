
import time
import random
import copy

import multiprocessing as mp

import numpy as np

from ..network import Topology, VertexSource, VertexSet, Task
from ..solver import PathSolver, FlowSolver, GreedySolver

from ..network import get_edge_length
from ..network.quantum import complete_swap, sequential_swap, HWParam

from ..utils.plot import plot_stack_bars, plot_optimized_network, plot_simple_topology



# evaluations
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

    node_budget = solver.node_budget_val
    edge_budget = solver.edge_budget_val

    Im = {node: 1 if solver.m[node].x > 0 else 0 for node in solver.m}
    Ic = {edge: 1 if solver.c[edge].x > 0 else 0 for edge in solver.c}
    
    return total_time, times, objs, node_budget, edge_budget, Im, Ic

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

    
    density = 5
    k = 100
    xs = ['1:10', '1:5', '1:1', '5:1', '10:1', '50:1', '100:1']
    ratios = [0.1, 0.2, 1, 5, 10, 50, 100]

    node_budgets = np.zeros((len(ratios), repeat)) * np.nan
    edge_budgets = np.zeros((len(ratios), repeat)) * np.nan

    vsrc = VertexSource.NOEL
    vset = VertexSet(vsrc)
    demand = 10
    task = Task(vset, 0.5, (demand, demand+1))
    for i, ratio in enumerate(ratios):
        print(f"ratio {ratio}")
        
        results = []
        pool = mp.Pool(repeat)

        for j in range(repeat):
            print(f"repeat {j}")

            params['pm'] = ratio
            params['pm_install'] = ratio*10
            params['pc'] = 1
            params['pc_install'] = 10

            temp_task = copy.deepcopy(task)
            net = Topology(task=temp_task, hw_params=params)
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
            total_time, times, objs, node_budget, edge_budget, Im, Ic = result.get()
            if node_budget is not None and edge_budget is not None:
                node_budgets[i, j] = node_budget
                edge_budgets[i, j] = edge_budget

        # use last solution for network topology
        # plot_simple_topology(net.graph, filename='./result/ratio/topology_original.png')
        users = set()
        for pair, demand in net.task.D.items():
            if demand > 0:
                users.add(pair[0])
                users.add(pair[1])
        plot_simple_topology(net.graph, Im, Ic, users, filename=f'./result/ratio/topology_{ratio}.png')

        # # average the results
        # avg_node_budgets = np.nanmean(node_budgets, axis=1)
        # avg_edge_budgets = np.nanmean(edge_budgets, axis=1)

        # # prepare for plotting
        # xs = ['1:10', '1:5', '1:1', '5:1', '10:1', ]
        # ys = {'Node Budget': avg_node_budgets, 'Edge Budget': avg_edge_budgets}

        # # plot
        # plot_stack_bars(
        #     xs, ys, 
        #     0.6, 
        #     # colors=['blue', 'green'],
        #     xlabel='Price Ratio', ylabel='Budget',
        #     filename=f'ratio.png'
        #     )
        # # percentage
        # avg_total_budgets = avg_node_budgets + avg_edge_budgets
        # ys = {'Node Budget': avg_node_budgets/avg_total_budgets, 'Edge Budget': avg_edge_budgets/avg_total_budgets}
        # plot_stack_bars(
        #     xs, ys, 
        #     0.6, 
        #     # colors=['blue', 'green'],
        #     xlabel='Price Ratio', ylabel='Budget Percentage',
        #     filename=f'ratio_percentage.png'
        #     )
        

if __name__ == '__main__':
    # set gurobi environment
    # mute parameter setting
    import gurobipy as gp
    gp.setParam('OutputFlag', 0)

    params = copy.deepcopy(HWParam)
    params['swap_prob'] = 0.75
    compare_price_ratio(params, repeat=1)
