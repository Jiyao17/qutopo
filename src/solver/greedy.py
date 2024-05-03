

from copy import deepcopy


import numpy as np
import networkx as nx



from ..network import VertexSet, VertexSource, Task, Topology, complete_swap, sequential_swap
from ..utils.plot import plot_nx_graph, plot_optimized_network


class GreedySolver():
    """
    greedy solver based on sequential path selection
    """
    def __init__(self, 
        network: Topology, 
        k: int=5, 
        weight: str='length',
        greedy_opt: str='resource',
        ) -> None:
        """
        network: Topology
            - the network to optimize
        k: int, optional (default=5)
            - the number of shortest paths to consider for each pair
        weight: str, optional (default='length')
            - the weight to consider for path selection
            - 'length': shortest path length
            - 'hop': least hops
        greedy_opt: str, optional (default='resource')
            - the optimization objective
            - 'resource': minimize resource usage
            - 'cost': minimize cost
        """

        self.network = network
        self.residual = deepcopy(network)
        self.k = k
        self.weight = weight
        self.greed_opt = greedy_opt

        self.U = network.U
        self.D = network.D

        self.paths = self.all_pairs_YenKSP(weight='length')
        self.obj_val = None
        self.add_variables()

    def all_pairs_YenKSP(self, weight=None):
        """
        find k shortest paths between all pairs in D
        weight: str, optional (default=None)
            -None: least hops
            -'length': Shortest path length
        """
        paths: 'dict[tuple[int], list[tuple[int]]]' = { pair: [] for pair in self.D.keys() }
        for pair in self.D.keys():
            src, dst = pair
            path_iter = nx.shortest_simple_paths(self.network.graph, src, dst, weight=weight)
            for _ in range(self.k):
                try:
                    path = tuple(next(path_iter))
                    paths[pair].append(path)
                except StopIteration:
                    break

        return paths

    def solve_path_resource(self, paths, swap_func: 'function' = complete_swap):
        """
        solve the paths for their resource usage
        """
        
        swap_prob = self.network.hw_params['swap_prob']
        
        # alpha[(u, p, e)] = # of entanglements used on edge e for pair u via path p
        alpha = {}
        for pair in self.D.keys():
            for path in paths[pair]:
                edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                
                costs = swap_func([1,] * len(edges), swap_prob)
                for i, edge in enumerate(edges):
                    alpha[(pair, path, edge)] = costs[i]

        # beta[(u, p, v)] = # of memory slots used at node v for pair u via path p
        beta = {}
        for pair in self.D.keys():
            for path in paths[pair]:
                for i, node in enumerate(path):
                    if i == 0:
                        beta[(pair, path, node)] = alpha[(pair, path, (node, path[1]))]
                    elif i == len(path) - 1:
                        beta[(pair, path, node)] = alpha[(pair, path, (path[-2], node))]
                    else:
                        mem_left = alpha[(pair, path, (path[i-1], node))]
                        mem_right = alpha[(pair, path, (node, path[i+1]))]
                        beta[(pair, path, node)] = mem_left + mem_right
                        
        return alpha, beta

    def solve_edge_cost(self, paths, alpha, beta):
        """
        solve the paths for their cost
        """
        edge_cost = {}
        dchannel = {}
        node_cost = {}
        for pair in self.D.keys():
            for path in paths[pair]:
                for edge in path:
                    channel_num = self.c[edge]
                    channel_cap = self.network.graph.edges[edge]['channel_capacity']
                    rem = channel_cap * channel_num - alpha[(pair, path, edge)]
                    if rem < 0:
                        dchannel[(pair, path, edge)] = int(np.ceil(abs(-rem) / channel_cap))
                        edge_cost[(pair, path, edge)] = dchannel
                    else:
                        edge_cost[(pair, path, edge)] = 0

                    if channel_num == 0 and dchannel > 0:
                        edge_cost[(pair, path, edge)] += self.network.hw_params['pc_install']

                for node in path:
                    dmem = beta[(pair, path, node)] 
                    node_cost[(pair, path, node)] = dmem * self.network.hw_params['pm']

                    mem = self.m[node]
                    if mem == 0 and dmem > 0:
                        node_cost[(pair, path, node)] += self.network.hw_params['pm_install']

        return edge_cost, dchannel, node_cost

    def add_variables(self):
        """
        add variables to the model
        """

        # x[(u, p)] generate x[(u, p)] entanglements for pair u using path p
        self.x = {}
        for u in self.D.keys():
            for p in self.paths[u]:
                # self.x[(u, p)] = self.model.addVar(vtype=gp.GRB.INTEGER, name=f'x_{u}_{p}')
                # no name to avoid too long name for gurobi
                self.x[(u, p)] = 0

        # c[e] number of channels used on edge e
        self.c = {}
        for edge in self.network.graph.edges:
            self.c[edge] = 0

        self.phi = {}
        for edge in self.network.graph.edges:
            self.phi[edge] = 0

        # m[v] number of memory slots used at node v
        self.m = {}
        # indicator variable for memory usage
        # 1 if memory is used, 0 otherwise
        nodes = self.network.graph.nodes(data=False)
        for node in nodes:
            self.m[node] = 0

    def calc_budget(self):
        """
        define the budgets for the resources
        """
        pm = self.network.hw_params['pm']
        pm_install = self.network.hw_params['pm_install']
        pc = self.network.hw_params['pc']
        pc_install = self.network.hw_params['pc_install']
        # memory budget
        self.pv = {}
        for node in self.network.graph.nodes(data=False):
            # per slot cost
            self.pv[node] = pm * self.m[node]
            # installation cost
            self.pv[node] += pm_install if self.m[node] > 0 else 0
        # edge budget
        edges = self.network.graph.edges(data=False)
        self.pe = {}
        for edge in edges:
            self.pe[edge] = pc * self.c[edge]
            self.pe[edge] += pc_install if self.c[edge] > 0 else 0

        self.budget = sum(self.pv.values()) + sum(self.pe.values())
        return self.budget

    def solve_resource(self):
        """
        solve the network for resource optimization
        """
        alpha, beta = self.solve_path_resource(self.paths)
        edge_cost, dchannel, node_cost = self.solve_edge_cost(self.paths, alpha, beta)

        for dpair in self.D.keys():
            best_path = None
            best_cost = np.inf
            for path in self.paths[dpair]:
                cost = sum([alpha[(dpair, path, edge)] for edge in path])
                if cost < best_cost:
                    best_path = path
                    best_cost = cost

            for edge in best_path:
                self.c[edge] += dchannel[(dpair, best_path, edge)]
                self.phi[edge] += alpha[(dpair, best_path, edge)]
                self.m
        




    def solve(self):
        pass