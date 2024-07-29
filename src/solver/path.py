
import random
import time

import numpy as np
import networkx as nx
import gurobipy as gp

from ..network.quantum import get_edge_length
from ..network import VertexSet, VertexSource, Task, Topology, VertexSetRandom, complete_swap, sequential_swap
from ..utils.plot import plot_nx_graph, plot_optimized_network
from ..utils.callback import callback


class PathSolver():
    """
    path augmentation-based
    """
    def __init__(self, 
        network: Topology, 
        k: int=100, 
        edge_weight: str=None,
        time_limit: int=60,
        mip_gap: float=0.01,
        output: bool=False
        ) -> None:

        self.network = network
        self.k = k # candidate path number between each SD pair
        self.edge_weight = edge_weight
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.output = output

        self.U = network.U
        self.D = network.D

        self.model = gp.Model()
        self.obj_val = None
        self.obj_vals = []
        self.node_budget_val = None
        self.edge_budget_val = None
        
        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPGap', mip_gap)
        if output is False:
            self.model.setParam('OutputFlag', 0)

        self.model._obj_vals = []

        # allow only single thread
        # self.model.setParam('Threads', 1)


    def prepare_paths(self, swap_func:callable=complete_swap, existing_paths: dict=None):
        if self.output:
            print("searching paths...")
        # shortest paths
        # half by length, half by hops
        # half_k = self.k // 2
        self.paths = self.all_pairs_YenKSP(weight='length', k=self.k, existing_paths=existing_paths)
        # shortest_hops_paths = self.all_pairs_YenKSP(weight=None, k=half_k, existing_paths=existing_paths)
        # self.paths = {}
        # for pair in self.D.keys():
        #     self.paths[pair] = shortest_length_paths[pair] + shortest_hops_paths[pair]
        
        if self.output:
            print("solving paths...")
        self.alpha, self.beta = self.solve_paths(self.paths, swap_func=swap_func)

    def build(self,):
        """
        build the model
        """
        if self.output:
            print("Building the model...")
        # find k shortest paths between all pairs in D
        # and pre-solve the paths
        if self.output:
            print("building linear model...")

        start = time.time()
        self.add_variables()
        end = time.time()
        if self.output:
            print(f"Variables added. Time elapsed: {end - start}")

        start = time.time()
        self.add_resource_constr()
        end = time.time()
        if self.output:
            print(f"Resource constraints added. Time elapsed: {end - start}")

        start = time.time()
        self.add_budget_def()
        end = time.time()
        if self.output:
            print(f"Budget definition added. Time elapsed: {end - start}")

        start = time.time()     
        self.add_demand_constr()
        end = time.time()
        if self.output:
            print(f"Demand constraints added. Time elapsed: {end - start}")

        if self.output:
            print("Model building done.")

    def all_pairs_YenKSP(self, weight=None, k=1, existing_paths: dict=None):
        """
        find k shortest paths between all pairs in D
        weight: str, optional (default=None)
            -None: least hops
            -'length': Shortest path length
        existing_paths: dict, optional (default=None)
        """
        paths: 'dict[tuple[int], list[tuple[str]]]' = { pair: [] for pair in self.D.keys() }
        
        if existing_paths is not None:
            for pair in existing_paths.keys():
                for i, path in enumerate(existing_paths[pair]):
                    if i <= k:
                        paths[pair].append(path)
        
        for pair in self.D.keys():
            src, dst = pair
            path_iter = nx.shortest_simple_paths(self.network.graph, src, dst, weight=weight)
            while len(paths[pair]) < k:
                try:
                    path = tuple(next(path_iter))
                    paths[pair].append(path)
                except StopIteration:
                    break

        return paths

    def solve_paths(self, paths, swap_func: 'function' = complete_swap):
        """
        solve the paths
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
                self.x[(u, p)] = self.model.addVar(vtype=gp.GRB.INTEGER)
                self.model.addConstr(self.x[(u, p)] >= 0)

        # c[e] number of channels used on edge e
        self.c = {}
        self.Ic = {}
        max_channel = 1e5
        for edge in self.network.graph.edges:
            self.c[edge] = self.model.addVar(vtype=gp.GRB.INTEGER, name=f'c_{edge}')
            self.model.addConstr(self.c[edge] >= 0)
            
            self.Ic[edge] = self.model.addVar(
                vtype=gp.GRB.BINARY,
                name=f'Ic_{edge}'
                    )
            # make Ic the indicator variable
            # Ic = 1 if c > 0, 0 otherwise
            self.model.addConstr(self.c[edge] <= max_channel * self.Ic[edge])
            self.model.addConstr(self.c[edge] >= self.Ic[edge])

        # 
        self.phi = {}
        for edge in self.network.graph.edges:
            self.phi[edge] = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS,
                name=f'phi_{edge}'
                )
            self.model.addConstr(self.phi[edge] >= 0)
            
        # m[v] number of memory slots used at node v
        self.m = {}
        # indicator variable for memory usage
        # 1 if memory is used, 0 otherwise
        self.Im = {} 
        max_mem = 1e5
        nodes = self.network.graph.nodes(data=False)
        for node in nodes:
            self.m[node] = self.model.addVar(
                vtype=gp.GRB.INTEGER,
                name=f'm_{node}'
                )
            self.model.addConstr(self.m[node] >= 0)

            self.Im[node] = self.model.addVar(
                vtype=gp.GRB.BINARY,
                name=f'Im_{node}'
                )
            self.model.addConstr(self.m[node] >= self.Im[node])
            self.model.addConstr(self.m[node] <= max_mem * self.Im[node])

    def add_demand_constr(self):
        """
        add demand constraints
        """
        for pair in self.D.keys():
            # demand constraint
            self.model.addConstr(
                gp.quicksum([self.x[(pair, path)] for path in self.paths[pair]]) 
                >= self.D[pair]
                )

    def add_resource_constr(self):
        """
        add resource constraints
        """
        # edge constraint
        for edge in self.network.graph.edges(data=False):
            phi = 0
            for pair in self.D.keys():
                for path in self.paths[pair]:
                    edges = tuple((path[i], path[i+1]) for i in range(len(path)-1))
                    if edge in edges:
                        phi += self.alpha[(pair, path, edge)] * self.x[(pair, path)]
                    elif (edge[1], edge[0]) in edges:
                        phi += self.alpha[(pair, path, (edge[1], edge[0]))] * self.x[(pair, path)]
            self.model.addConstr(self.phi[edge] >= phi)

        # channel constraint
        for edge in self.network.graph.edges(data=False):
            channel_capacity = self.network.graph.edges[edge]['channel_capacity']
            self.model.addConstr(channel_capacity * self.c[edge] >= self.phi[edge])

        # memory constraint
        m = { node: 0 for node in self.network.graph.nodes(data=False)}
        for edge in self.network.graph.edges(data=False):
            u, v = edge
            m[u] += self.phi[edge]
            m[v] += self.phi[edge]
        for node in self.network.graph.nodes(data=False):
            self.model.addConstr(self.m[node] >= m[node])

    def add_budget_def(self):
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
            self.pv[node] += pm_install * self.Im[node]
        # edge budget
        edges = self.network.graph.edges(data=False)
        self.pe = {}
        for edge in edges:
            length = self.network.graph.edges[edge]['length']
            self.pe[edge] = pc * self.c[edge] * length
            self.pe[edge] += pc_install * self.Ic[edge]

        self.node_budget = gp.quicksum(self.pv.values())
        self.edge_budget = gp.quicksum(self.pe.values())
        self.budget = self.node_budget + self.edge_budget

    def solve(self, callback=callback):
        
        self.model.setObjective(self.budget, gp.GRB.MINIMIZE)

        self.model.optimize(callback)

        try:
            self.obj_val = self.model.objVal
            self.obj_vals = self.model._obj_vals

            self.node_budget_val = self.node_budget.getValue()
            self.edge_budget_val = self.edge_budget.getValue()
        except AttributeError:
            self.obj_val = None
            self.obj_vals = []

            self.node_budget_val = None
            self.edge_budget_val = None
            print("Path model is not solved.")


class PathSolverNonCost(PathSolver):
    def add_resource_constr(self):
        """
        add resource constraints
        """
        # edge constraint
        for edge in self.network.graph.edges(data=False):
            phi = 0
            for pair in self.D.keys():
                for path in self.paths[pair]:
                    edges = tuple((path[i], path[i+1]) for i in range(len(path)-1))
                    if edge in edges:
                        phi += self.alpha[(pair, path, edge)] * self.x[(pair, path)]
                    elif (edge[1], edge[0]) in edges:
                        phi += self.alpha[(pair, path, (edge[1], edge[0]))] * self.x[(pair, path)]
            self.model.addConstr(self.phi[edge] >= phi)

        # channel constraint
        for edge in self.network.graph.edges(data=False):
            channel_capacity = int(self.network.graph.edges[edge]['channel_capacity'])
            self.model.addConstr(self.phi[edge] <= channel_capacity * self.c[edge])
            self.model.addConstr(self.phi[edge] >= channel_capacity * (self.c[edge] - 1))

        # memory constraint
        m = { node: 0 for node in self.network.graph.nodes(data=False)}
        for edge in self.network.graph.edges(data=False):
            u, v = edge
            m[u] += self.phi[edge]
            m[v] += self.phi[edge]
        for node in self.network.graph.nodes(data=False):
            self.model.addConstr(self.m[node] >= m[node])
            self.model.addConstr(self.m[node] <= m[node] + 1)

    def solve(self):
        """
        find a feasible solution without cost minimization
        """
        self.model.setObjective(0, gp.GRB.MINIMIZE)

        self.model.optimize()

        try:
            self.obj_val = self.budget.getValue()
        except AttributeError:
            print("Model is not solved.")
            print("Probably infeasible or unbounded.")


class PathSolverMinResource(PathSolverNonCost):

    def solver(self):
        """
        find a feasible solution with minimum resource usage
        """
        self.model.setObjective(gp.quicksum(self.m.values()) + gp.quicksum(self.c.values()), gp.GRB.MINIMIZE)

        self.model.optimize()

        try:
            self.obj_val = self.model.objVal
        except AttributeError:
            print("Model is not solved.")
            print("Probably infeasible or unbounded.")



if __name__ == "__main__":
    vsrc = VertexSource.EENET
    vset = VertexSet(vsrc)

    # vset = VertexSetRandom(10)
    # vset.scale()

    demand = 10
    task = Task(vset, 0.5, (demand, demand+1))
    net = Topology(task=task)

    net.make_clique(list(net.graph.nodes(data=False)))
    # find longest edge
    edge = max(net.graph.edges(data='length'), key=lambda x: x[2])
    print(edge)


    seg_len = get_edge_length(demand, net.hw_params['photon_rate'], net.hw_params['fiber_loss'])

    net.connect_nodes_nearest(6)
    # make sure the network is connected
    net.connect_nearest_component()
    net.segment_edges(seg_len, seg_len)


    net.plot(None, None, './result/path/fig.png')

    # print(net.graph.edges(data=True))

    k = 100
    start = time.time()
    solver = PathSolver(net, k, mip_gap=0.05, output=True)
    # solver = PathSolverNonCost(net, k, output=True)
    # solver = PathSolverMinResource(net, k, output=True)
    solver.prepare_paths(swap_func=sequential_swap)
    solver.build()
    solver.solve()
    end = time.time()
    print(f"Time elapsed: {end - start}")
    print("Objective value: ", solver.obj_vals)



