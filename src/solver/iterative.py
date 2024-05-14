

import random
import numpy as np

import matplotlib.pyplot as plt
import geopy.distance as geo

from ..network.quantum import get_edge_length
from ..network import VertexSet, VertexSource, Task, Topology
from .path import PathSolver
from ..utils.plot import plot_optimized_network


class IterSolver:

    def __init__(self, net: Topology, SolverClass=PathSolver):
        self.SolverClass = SolverClass
        self.net = net

        self.solver: PathSolver = None

    def add_nodes(self, rd: int=1, demand: int=200):
        """
        find new possible nodes using
            -network
            -optimal solution
        """

        # add random nodes
        # city_num = len(self.net.U.items())
        # self.net.add_nodes_random(city_num, rd)
        # self.net.add_nodes_random_on_edges(rd)
        seg_len = get_edge_length(demand, self.net.hw_params['photon_rate'], self.net.hw_params['fiber_loss'])
        
        if rd == 1:
            self.net.connect_nodes_nearest(10, rd)
            self.net.connect_nearest_component(rd)
            # self.net.make_clique(list(self.net.graph.nodes), 1)
            # seg_len = get_edge_length(demand, self.net.hw_params['photon_rate'], self.net.hw_params['fiber_loss'])
            self.net.segment_edges(seg_len, seg_len, rd)
            # self.net.connect_nodes_radius(200, rd)
            return
        
        # fully random nodes
        self.net.add_nodes_random(len(self.net.task.U), rd)
        
        # add around used nodes
        # used_nodes = [node for node in self.net.graph.nodes if self.solver.m[node].x > 0]
        # do not add around original nodes
        # used_nodes = [node for node in used_nodes if node not in self.net.U]
        # self.net.add_nodes_random_nearby(used_nodes, 20, 1, rd)
        # self.net.add_nodes_random_nearby(used_nodes, 20, 3, rd)

        indices = np.arange(len(self.net.pairs))
        pair_num = len(self.net.U.items())
        pairs = [self.net.pairs[i] for i in np.random.choice(indices, pair_num, replace=False)]
        for pair in pairs:
            if not self.net.graph.has_edge(*pair):
                pos_u = self.net.graph.nodes[pair[0]]['pos']
                pos_v = self.net.graph.nodes[pair[1]]['pos']
                length = geo.geodesic(pos_u, pos_v).km
                self.net.graph.add_edge(*pair, length=length, group=rd)
            else:
                length = self.net.graph.edges[pair]['length']

            seg_num = int(length / seg_len)
            self.net.segment_edge_line(*pair, seg_num, rd)

        self.net.update_edges()
        self.net.plot(None, None, f'./result/iterative/fig_random_{rd}.png')
        
        # self.net.connect_nodes_nearest(5, rd)
        # self.net.plot(None, None, f'./result/iterative/fig_nearest_{rd}.png')
        self.net.connect_nodes_radius(seg_len*1.5, rd)
        # self.net.connect_nearest_component(rd)
        # self.net.segment_edges(150, 150, rd)
        # self.net.plot(None, None, f'./result/iterative/fig_cluster_{rd}.png')

        # net.plot(None, None, f'./result/iterative/fig_segment_{rd}.png')

    def del_nodes(self, rd: int=1):
        """
        remove unlikely nodes using
            -network
            -optimal solution
        """
        nodes = list(self.net.graph.nodes(data=True))
        self.best_obj = np.inf
        self.best_nodes = []
        # update best nodes
        if self.solver.obj_val < self.best_obj:
            self.best_obj = self.solver.obj_val
            self.best_nodes = [node for node, data in nodes if int(self.solver.m[node].x) > 0]

        for node, data in nodes:
            m = int(self.solver.m[node].x)
            # mark utilized nodes
            if 'marked' not in data:
                data['marked'] = False
            if m > 0:
                data['marked'] = True

            # remove non-original nodes
            if data['group'] != 0 and node not in self.best_nodes:
                # not used in this round 
                if m == 0:
                    # never used before
                    if data['marked'] == False:
                        self.net.graph.remove_node(node)
                    # used before but expired
                    elif data['group'] <= rd - 5:
                        self.net.graph.remove_node(node)
            
        # net.cluster_inter_nodes(city_num, set([i]))
        # net.plot(None, None, f'./result/iterative/fig_merge_{rd}.png')

        # self.net.connect_nearest_component(rd)

    def iterative_solve(self, round_num: int=10, k: int=10, demand: int=200):
        """
        iterative solve
        """
        past_rounds = set()
        vals = []
        self.net.plot(None, None, f'./result/iterative/fig_init.png')
        for i in range(1, round_num + 1):
            self.add_nodes(i, demand)
            # else:
            #     self.net.make_clique(list(self.net.graph.nodes), 1)
            #     self.net.segment_edges(150, 150, 1)
                
            self.net.plot(None, None, f'./result/iterative/fig_add_{i}.png')

            # nodes = list(self.net.G.nodes(data=True))
            # for node in nodes:
            #     lat, lon = node[1]['pos']
            #     lat_min, lat_max = self.net.area['lat_min'], self.net.area['lat_max']
            #     lon_min, lon_max = self.net.area['lon_min'], self.net.area['lon_max']
            #     assert lat_min <= lat <= lat_max
            #     assert lon_min <= lon <= lon_max
            # self.net.connect_nearest_component(i)
            # self.solver = self.SolverClass(self.net, 10, 'length', output=False)
            self.solver = self.SolverClass(self.net, k, output=False)

            try:
                self.solver.solve()
            except AttributeError:
                print(f"Round {i} objective value: None.")
                self.solver.obj_val = np.nan
            else:
                print(f"Round {i} objective value: {self.solver.obj_val}.")
                m = {node: int(self.solver.m[node].x) for node in self.solver.m}
                c = {edge: int(self.solver.c[edge].x) for edge in self.solver.c}
                phi = {edge: self.solver.phi[edge].x for edge in self.solver.phi}
                plot_optimized_network(
                    self.solver.network.graph, 
                    m, c, phi,
                    filename=f'./result/iterative/fig_solved_{i}.png'
                    )
                
                self.del_nodes(i)

                self.net.plot(None, None, f'./result/iterative/fig_del_{i}.png')
            finally:
                past_rounds.add(i)
                vals.append(self.solver.obj_val)

                # plot vals v.s. rounds
                rds = list(past_rounds)
                rds.sort()
                plt.plot(rds, vals)
                plt.xlabel('rounds')
                plt.ylabel('objective value')
                plt.savefig('./result/iterative/fig_vals.png')

                plt.close()



if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    vsrc = VertexSource.TRIANGLE
    vset = VertexSet(vsrc)
    demand = 100
    task = Task(vset, 1.0, (demand, demand+1))
    net = Topology(task=task)
    city_num = len(net.graph.nodes)

    # net.connect_nodes_nearest(10, 1)
    # net.segment_edges(200, 200, 1)
    # net.connect_nodes_radius(200, 1)
    # net.connect_nearest_component(1)
    net.plot(None, None, './result/iterative/fig.png')


    # solver = PathSolverNonCost(net, k, output=True)
    # solver = PathSolverMinResource(net, k, output=True)
    # solver.solve()

    iter_solver = IterSolver(net, PathSolver)
    iter_solver.iterative_solve(10, k=20, demand=demand)

    
    # print("Objective value: ", iter_solver.solver.obj_val)
    # plot_optimized_network(
    #     iter_solver.solver.network.G, 
    #     solver.m, solver.c, solver.phi,
    #     filename='./result/path/fig-solved.png'
    #     )



