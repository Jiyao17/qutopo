

import random
import numpy as np

import matplotlib.pyplot as plt

from ..network import VertexSet, VertexSource, Task, Topology
from .path import PathSolver
from ..utils.plot import plot_optimized_network


class IterSolver:

    def __init__(self, net: Topology, Solver: object=PathSolver):
        self.Solver = Solver
        self.net = net

        net.connect_nearest_component(1)
        self.solver: PathSolver = PathSolver(self.net, 20, 'length', output=False)

        

    def add_nodes(self, rd: int=1):
        """
        find new possible nodes using
            -network
            -optimal solution
        """

        # add random nodes
        # self.net.add_nodes_random(len(self.net.G.nodes)*3, rd)
        # self.net.update_edges()
        # self.net.plot(None, None, f'./result/path/fig_random_{rd}.png')
        
        self.net.connect_nodes_nearest(10, rd)
        self.net.plot(None, None, f'./result/path/fig_nearest_{rd}.png')
        self.net.segment_edges(200, 200, rd)
        self.net.connect_nodes_radius(200, rd)
        self.net.connect_nearest_component(rd)
        # self.net.plot(None, None, f'./result/path/fig_cluster_{rd}.png')

        # net.plot(None, None, f'./result/path/fig_segment_{rd}.png')

    def del_nodes(self, rd: int=1):
        """
        remove unlikely nodes using
            -network
            -optimal solution
        """
        nodes = list(self.net.G.nodes(data=True))
        for node, data in nodes:
            m = int(self.solver.m[node].x)
            if m == 0 and data['group'] <= rd and data['group'] != 0:
                self.net.G.remove_node(node)
            
        # net.cluster_inter_nodes(city_num, set([i]))
        # net.plot(None, None, f'./result/path/fig_merge_{rd}.png')

        # self.net.connect_nearest_component(rd)

    def iterative_solve(self, round_num: int=10):
        """
        iterative solve
        """
        past_rounds = set()
        vals = []
        for i in range(1, round_num + 1):
            
            self.add_nodes(i)
            self.net.plot(None, None, f'./result/path/fig_add_{i}.png')

            # nodes = list(self.net.G.nodes(data=True))
            # for node in nodes:
            #     lat, lon = node[1]['pos']
            #     lat_min, lat_max = self.net.area['lat_min'], self.net.area['lat_max']
            #     lon_min, lon_max = self.net.area['lon_min'], self.net.area['lon_max']
            #     assert lat_min <= lat <= lat_max
            #     assert lon_min <= lon <= lon_max
            
            self.solver.build()
            self.solver.solve()

            print(f"Round {i} objective value: {self.solver.obj_val}.")
            plot_optimized_network(
                self.solver.network.G, 
                self.solver.m, self.solver.c, self.solver.phi,
                filename=f'./result/path/fig_solved_{i}.png'
                )

            self.del_nodes(i)
            self.net.plot(None, None, f'./result/path/fig_del_{i}.png')

            past_rounds.add(i)
            vals.append(self.solver.obj_val)

            # plot vals v.s. rounds
            rds = list(past_rounds)
            rds.sort()
            plt.plot(rds, vals)
            plt.xlabel('rounds')
            plt.ylabel('objective value')
            plt.savefig('./result/path/fig_vals.png')








if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    vsrc = VertexSource.NOEL
    vset = VertexSet(vsrc)
    task = Task(vset, 1.0, (100, 101))
    net = Topology(task=task)
    city_num = len(net.G.nodes)

    # solver = PathSolverNonCost(net, k, output=True)
    # solver = PathSolverMinResource(net, k, output=True)
    # solver.solve()

    iter_solver = IterSolver(net, PathSolver)
    iter_solver.iterative_solve(100)

    
    # print("Objective value: ", iter_solver.solver.obj_val)
    # plot_optimized_network(
    #     iter_solver.solver.network.G, 
    #     solver.m, solver.c, solver.phi,
    #     filename='./result/path/fig-solved.png'
    #     )



