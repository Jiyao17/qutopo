
import random
import time

import numpy as np
import networkx as nx
import gurobipy as gp

from ..network.quantum import get_edge_length
from ..network import VertexSet, VertexSource, Task, Topology, complete_swap, sequential_swap
from ..utils.plot import plot_nx_graph, plot_optimized_network


class PathSolver():
    """
    path augmentation-based
    """
    def __init__(self, 
        network: Topology, 
        k: int=5, 
        edge_weight: str=None,
        time_limit: int=60,
        mip_gap: float=0.01,
        output: bool=False
        ) -> None:

        self.network = network
        self.k = k
        self.edge_weight = edge_weight
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.output = output

        self.U = network.U
        self.D = network.D

        self.model = gp.Model()
        self.obj_val = None
        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPGap', mip_gap)
        if output is False:
            self.model.setParam('OutputFlag', 0)

        self.build(edge_weight=edge_weight)

    def build(self, edge_weight=None):
        """
        build the model
        """
        if self.output:
            print("Building the model...")
        # find k shortest paths between all pairs in D
        # and pre-solve the paths
        if self.output:
            print("searching paths...")
        self.paths = self.all_pairs_YenKSP(weight=edge_weight)

        if self.output:
            print("solving paths...")
        self.alpha, self.beta = self.solve_paths(self.paths)

        if self.output:
            print("building linear model...")

        self.add_variables()
        self.add_resource_constr()
        self.add_budget_def()
        self.add_demand_constr()
        
        if self.output:
            print("Model building done.")

    def all_pairs_YenKSP(self, weight=None):
        """
        find k shortest paths between all pairs in D
        weight: str, optional (default=None)
            -None: least hops
            -'length': Shortest path length
        """
        paths: 'dict[tuple[int], list[tuple[str]]]' = { pair: [] for pair in self.D.keys() }
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

        self.budget = gp.quicksum(self.pv.values()) + gp.quicksum(self.pe.values())

    def solve(self):
        
        self.model.setObjective(self.budget, gp.GRB.MINIMIZE)

        self.model.optimize()

        try:
            self.obj_val = self.model.objVal
        except AttributeError:
            print("Model is not solved.")
            print("Probably infeasible or unbounded.")
            raise AttributeError



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
    seed = 66
    random.seed(seed)
    np.random.seed(seed)

    vsrc = VertexSource.NOEL
    vset = VertexSet(vsrc)
    demand = 100
    task = Task(vset, 1.0, (demand, demand+1))
    net = Topology(task=task)

    city_num = len(net.graph.nodes)
    seg_len = get_edge_length(demand, net.hw_params['photon_rate'], net.hw_params['fiber_loss'])
    print(f"Suggested edge length: {seg_len}")

    # net.connect_nodes_nearest(10, 1) # ~7.8e6
    # net.connect_nearest_component(1)
    # k=50    k=100   k=150   k=200   k=500
    # ~3.5e7  ~3.2e7  ~2.8e7  ~1e7    ~6.4e6
    net.make_clique(list(net.graph.nodes), 1) 
    net.segment_edges(seg_len, seg_len, 1)
    # net.segment_edges(100, 100, 1)
    # net.connect_nodes_radius(200, 1)

    net.plot(None, None, './result/path/fig.png')

    # print(net.graph.edges(data=True))

    k = 1000
    start = time.time()
    solver = PathSolver(net, k, output=True)
    # solver = PathSolverNonCost(net, k, output=True)
    # solver = PathSolverMinResource(net, k, output=True)
    solver.solve()
    end = time.time()
    print(f"Time elapsed: {end - start}")

    m = { node: int(m.x) for node, m in solver.m.items() }
    c = { edge: int(c.x) for edge, c in solver.c.items() }
    phi = { edge: phi.x for edge, phi in solver.phi.items() }
    print("Objective value: ", solver.obj_val)
    plot_optimized_network(
        solver.network.graph,
        m, c, phi,
        False,
        filename='./result/path/fig-solved.png'
        )



