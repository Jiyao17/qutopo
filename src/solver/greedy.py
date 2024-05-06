

from copy import deepcopy
import random

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
        k: int=10, 
        weight: str=None,
        greedy_opt: str='cost',
        ) -> None:
        """
        network: Topology
            - the network to optimize
        k: int, optional (default=5)
            - the number of shortest paths to consider for each pair
        weight: str, optional (default='length')
            - the weight to consider for path selection
            - 'length': shortest path length
            - None: least hops
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

        self.obj_val = None
        self.paths = self.all_pairs_YenKSP(weight='length')
        self.add_variables()
        self.alpha, self.beta = self.solve_paths(self.paths)

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

            if len(paths[pair]) == 0:
                raise ValueError(f'No path found between {src} and {dst}')

        return paths

    def solve_paths(self, paths, swap_func: 'function' = complete_swap):
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

    def try_path(self, path, demand):
        """
        solve the paths for their cost
        path: path to try
        demand: demand for the path
        alpha: edge cost
        beta: node cost
        """
        pair = (path[0], path[-1])
        
        dchannels = {}
        dmems = {}
        edge_cost = {}
        node_cost = {}

        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        for edge in edges:
            channel_num = self.c[edge] if edge in self.c else self.c[(edge[1], edge[0])]
            channel_cap = self.network.graph.edges[edge]['channel_capacity']
            edge_length = self.network.graph.edges[edge]['length']
            rem = channel_cap * channel_num - self.alpha[(pair, path, edge)] * demand
            if rem < 0:
                dchannels[edge] = int(np.ceil(abs(rem) / channel_cap))
                edge_cost[edge] = dchannels[edge] * self.network.hw_params['pc'] * edge_length
            else:
                dchannels[edge] = 0
                edge_cost[edge] = 0

            if channel_num == 0 and dchannels[edge] > 0:
                edge_cost[edge] += self.network.hw_params['pc_install']

        for node in path:
            dmems[node] = self.beta[(pair, path, node)] * demand
            node_cost[node] = dmems[node] * self.network.hw_params['pm']

            if self.m[node] == 0 and dmems[node] > 0:
                node_cost[node] += self.network.hw_params['pm_install']

        return dchannels, dmems, edge_cost, node_cost

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
            length = self.network.graph.edges[edge]['length']
            self.pe[edge] = pc * self.c[edge] * length
            self.pe[edge] += pc_install if self.c[edge] > 0 else 0

        self.budget = sum(self.pv.values()) + sum(self.pe.values())
        return self.budget

    def optimize(self, criterion='resource', ddemand=1):
        """
        solve the network for resource optimization
        criterion: str, optional (default='resource')
            - the optimization criterion
            - 'resource': minimize resource usage
            - 'cost': minimize cost
        ddemand: int, optional (default=1)
            - the demand to optimize each time
        """
        pairs = list(self.D.keys())
        demands = list(self.D.values())

        i = -1
        while max(demands) > 0:
            i = (i + 1) % len(pairs)
            dpair = pairs[i]
            demands[i] -= ddemand
            demand = ddemand

            best_path = None
            best_resource = np.inf
            best_cost = np.inf
            if criterion == 'resource':
                for path in self.paths[dpair]:
                    edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                    resource = sum([self.alpha[(dpair, path, edge)] for edge in edges])
                    if resource < best_resource:
                        best_path = path
                        best_resource = resource
            elif criterion == 'cost':
                for path in self.paths[dpair]:
                    dchannels, dmems, edge_cost, node_cost = self.try_path(path, demand)
                    cost = sum(edge_cost.values()) + sum(node_cost.values())
                    if cost < best_cost:
                        best_path = path
                        best_cost = cost
            else:
                raise NotImplementedError(f'criterion {criterion} not implemented')
            
            # analyze the best path
            dchannels, dmems, edge_cost, node_cost = self.try_path(best_path, demand)
            # finalize the path
            self.x[(dpair, best_path)] += demand
            edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)]
            for edge in edges:
                if edge in self.c:
                    self.c[edge] += dchannels[edge] if edge in dchannels else dchannels[(edge[1], edge[0])]
                    self.phi[edge] += self.alpha[(dpair, best_path, edge)]
                else:
                    self.c[(edge[1], edge[0])] += dchannels[edge] if edge in dchannels else dchannels[(edge[1], edge[0])]
                    self.phi[(edge[1], edge[0])] += self.alpha[(dpair, best_path, edge)]
            for node in best_path:
                self.m[node] += dmems[node]
        
    def solve(self):
        
        if self.greed_opt == 'resource':
            self.optimize('resource')
        elif self.greed_opt == 'cost':
            self.optimize('cost')

        self.obj_val = self.calc_budget()


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    vsrc = VertexSource.NOEL
    vset = VertexSet(vsrc)
    task = Task(vset, 1.0, (100, 101))
    net = Topology(task=task)

    city_num = len(net.graph.nodes)
    net.connect_nodes_nearest(5, 1)
    net.connect_nodes_radius(200, 1)
    net.connect_nearest_component(1)
    net.segment_edges(200, 200, 1)
    net.plot(None, None, './result/greedy/fig.png')

    k = 20
    # solver = GreedySolver(net, k, None, 'resource')
    solver = GreedySolver(net, k, None, 'cost')
    solver.solve()

    
    print("Objective value: ", solver.obj_val)
    plot_optimized_network(
        solver.network.graph, 
        solver.m, solver.c, solver.phi,
        filename='./result/greedy/fig-solved.png'
        )

