
import networkx as nx
import gurobipy as gp

from ..network import VertexSet, VertexSource, Task, Network
from ..utils.plot import plot_nx_graph, plot_optimized_network


class PathAugSolver():
    """
    path augmentation-based
    """
    def __init__(self, 
        network: Network, 
        k: int=3, 
        edge_weight: str=None,
        output: bool=False) -> None:

        self.network = network
        self.k = k
        self.edge_weight = edge_weight
        self.output = output

        self.U = network.U
        self.D = network.D

        self.model = gp.Model()
        self.obj_val = None
        if output is False:
            self.model.setParam('OutputFlag', 0)

        self.build(edge_weight=edge_weight)

    def build(self, edge_weight=None):
        """
        build the model
        """
        # find k shortest paths between all pairs in D
        # and pre-solve the paths
        self.paths = self.all_pairs_YenKSP(weight=edge_weight)
        self.alpha, self.beta = self.solve_paths()

        self.add_variables()
        self.add_resource_constr()
        self.add_budget_def()
        self.add_demand_constr()
        self.solve()

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
            path_iter = nx.shortest_simple_paths(self.network.G, src, dst, weight=weight)
            for _ in range(self.k):
                try:
                    path = tuple(next(path_iter))
                    paths[pair].append(path)
                except StopIteration:
                    break

        return paths

    def solve_paths(self):
        """
        solve the paths
        """
        # alpha[(u, p, e)] = # of entanglements used on edge e for pair u via path p
        alpha = {}
        for pair in self.D.keys():
            for path in self.paths[pair]:
                edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                for edge in edges:
                    alpha[(pair, path, edge)] = 1

        # beta[(u, p, v)] = # of memory slots used at node v for pair u via path p
        beta = {}
        for pair in self.D.keys():
            for path in self.paths[pair]:
                for node in path:
                    beta[(pair, path, node)] = 2

        return alpha, beta

    def add_variables(self):
        """
        add variables to the model
        """

        # x[(u, p)] generate x[(u, p)] entanglements for pair u using path p
        self.x = {}
        for u in self.D.keys():
            for p in self.paths[u]:
                self.x[(u, p)] = self.model.addVar(vtype=gp.GRB.INTEGER, name=f'x_{u}_{p}')
                self.model.addConstr(self.x[(u, p)] >= 0)

        # c[e] number of channels used on edge e
        self.c = {}
        self.Ic = {}
        max_channel = 1e5
        for edge in self.network.G.edges:
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
        for edge in self.network.G.edges:
            self.phi[edge] = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS,
                name=f'C_{edge}'
                )
            self.model.addConstr(self.phi[edge] >= 0)
            
        # m[v] number of memory slots used at node v
        self.m = {}
        # indicator variable for memory usage
        # 1 if memory is used, 0 otherwise
        self.Im = {} 
        max_mem = 1e5
        nodes = self.network.G.nodes(data=False)
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
                gp.quicksum(self.x[(pair, path)] for path in self.paths[pair]) 
                >= self.D[pair]
                )

    def add_resource_constr(self):
        """
        add resource constraints
        """
        # memory constraint
        for node in self.network.G.nodes(data=False):
            m = 0
            for pair in self.D.keys():
                for path in self.paths[pair]:
                    if node in path:
                        m += self.beta[(pair, path, node)] * self.x[(pair, path)]
            self.model.addConstr(self.m[node] >= m)


        # edge constraint
        for edge in self.network.G.edges(data=False):
            phi = 0
            for pair in self.D.keys():
                for path in self.paths[pair]:
                    edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                    if edge in edges:
                        phi += self.alpha[(pair, path, edge)] * self.x[(pair, path)]
            self.model.addConstr(self.phi[edge] >= phi)

        for edge in self.network.G.edges(data=False):
            channel_capacity = self.network.G.edges[edge]['channel_capacity']
            self.model.addConstr(self.phi[edge] <= self.c[edge] * channel_capacity)

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
        for node in self.network.G.nodes(data=False):
            # per slot cost
            self.pv[node] = pm * self.m[node]
            # installation cost
            self.pv[node] += pm_install * self.Im[node]
        # edge budget
        edges = self.network.G.edges(data=False)
        self.pe = {}
        for edge in edges:
            self.pe[edge] = pc * self.c[edge]
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


if __name__ == "__main__":
    vsrc = VertexSource.NOEL
    vset = VertexSet(vsrc)
    task = Task(vset, 0.2, (10, 11))
    net = Network(task=task)
    node_num = len(net.G.nodes)

    net.cluster_by_nearest(5)
    net.update_edges()
    net.plot(None, None, './result/fig_cluster.png')

    net.segment_edge(150, 150)
    net.plot(None, None, './result/fig_seg.png')

    net.update_pairs()
    net.update_edges()


    print(len(net.G.nodes), len(net.G.edges))
    net.plot(None, None, './result/fig.png')

    solver = PathAugSolver(net, 3)
    solver.solve()
    
    print("Objective value: ", solver.obj_val)
    plot_optimized_network(solver.network.G, solver.m, solver.c, filename='./result/fig-solved.png')



