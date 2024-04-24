
import networkx as nx
import gurobipy as gp

from ..network import VertexSet, VertexSource, Task, Network
from ..utils.plot import plot_nx_graph, plot_optimized_network


class FlowSolver():
    """
    solve the linear optimization problem
    """
    def __init__(self, 
            network: Network, 
            time_limit: int=100,
            relaxed: bool=False, 
            output: bool=False
            ) -> None:
        """
        if relaxed, all variables are continuous, and installation costs are not considered
        """

        self.network = network
        self.time_limit = time_limit
        self.relaxed = relaxed
        self.output = output

        self.model = gp.Model("NetworkConstruction")
        self.model.setParam('TimeLimit', time_limit)
        if output is False:
            self.model.setParam('OutputFlag', 0)
        self.obj_val = None
        self.build(relaxed)

    def build(self, relaxed: bool=False):
        """
        build the linear model
        """
        self.add_variables(relaxed)
        self.add_distribution_constr()
        self.add_resource_constr()
        self.add_budget_def()

    def add_variables(self, relaxed: bool=False):
        """
        add m, c, and f variables to the model
        """
        device_num_type = gp.GRB.INTEGER if not relaxed else gp.GRB.CONTINUOUS

        # add node memory variables
        self.m = {}
        # indicator variable for memory usage
        # 1 if memory is used, 0 otherwise
        self.Im = {} 
        max_mem = 1e4
        nodes = self.network.G.nodes(data=False)
        for node in nodes:
            self.m[node] = self.model.addVar(
                vtype=device_num_type,
                name=f'm_{node}'
                )
            self.model.addConstr(self.m[node] >= 0)

            if not relaxed:
                self.Im[node] = self.model.addVar(
                    vtype=gp.GRB.BINARY,
                    name=f'Im_{node}'
                    )
            # make Im the indicator variable
            # Im = 1 if m > 0, 0 otherwise
            if not relaxed:
                self.model.addConstr(self.m[node] >= self.Im[node])
                self.model.addConstr(self.m[node] <= max_mem * self.Im[node])
         
        # add edge channel variables
        self.c = {}
        self.Ic = {}
        max_channel = 1e2
        edges = self.network.G.edges(data=False)
        for edge in edges:
            self.c[edge] = self.model.addVar(
                vtype=device_num_type,
                name=f'c_{edge}'
                )
            self.model.addConstr(self.c[edge] >= 0)

            if not relaxed:
                self.Ic[edge] = self.model.addVar(
                    vtype=gp.GRB.BINARY,
                    name=f'Ic_{edge}'
                    )
            # make Ic the indicator variable
            # Ic = 1 if c > 0, 0 otherwise
            if not relaxed:
                self.model.addConstr(self.c[edge] <= max_channel * self.Ic[edge])
                self.model.addConstr(self.c[edge] >= self.Ic[edge])
        # add edge capacity variables
        self.phi = {} # \phi in the paper
        for edge in edges:
            self.phi[edge] = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS,
                name=f'C_{edge}'
                )
            self.model.addConstr(self.phi[edge] >= 0)
            
        pairs = self.network.pairs
        self.f = {}
        # add flow variables
        for out_pair in pairs:
            for  v in nodes:
                if v not in out_pair:
                    # (i, v) -> (i, j)
                    in_pair = (out_pair[0], v) if (out_pair[0], v) in pairs else (v, out_pair[0])
                    self.f[(out_pair, in_pair)] = self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        name=f'f_{out_pair}_{in_pair}'
                        )
                    self.model.addConstr(self.f[(out_pair, in_pair)] >= 0)
                    # (v, j) -> (i, j)
                    in_pair = (v, out_pair[1]) if (v, out_pair[1]) in pairs else (out_pair[1], v)
                    self.f[(out_pair, in_pair)] = self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        name=f'f_{out_pair}_{in_pair}'
                        )
            # for in_pair in pairs:
            #     if out_pair != in_pair:
            #         self.f[(out_pair, in_pair)] = self.model.addVar(
            #             vtype=gp.GRB.CONTINUOUS,
            #             name=f'f_{out_pair}_{in_pair}'
            #             )
            #         self.model.addConstr(self.f[(out_pair, in_pair)] >= 0)

    def add_distribution_constr(self):
        """
        add the constraints to the model
        """
        # definition of out-flows
        self.O = {}
        nodes = self.network.G.nodes(data=False)
        pairs = self.network.pairs
        for out_pair in pairs:
            i, j = out_pair
            self.O[(i, j)] = 0
            for v in nodes:
                if v != i and v != j:
                    in_pair1 = (i, v) if (i, v) in pairs else (v, i)
                    in_pair2 = (v, j) if (v, j) in pairs else (j, v)
                    self.O[out_pair] += self.f[(out_pair, in_pair1)] + self.f[(out_pair, in_pair2)]

        # definition (constraint) of in-flows
        self.I = {}
        swap_prob = self.network.hw_params['swap_prob']
        for in_pair in pairs:
            i, j = in_pair
            self.I[in_pair] = 0
            # entanglement generation
            # edge_num = self.network.G.number_of_edges(i, j)
            # for key in range(edge_num):
            #     self.I[in_pair] += self.phi[(i, j, key)] if (i, j, key) in self.phi else self.phi[(j, i, key)]
            self.I[in_pair] += self.phi[(i, j)] if (i, j) in self.phi else 0
            self.I[in_pair] += self.phi[(j, i)] if (j, i) in self.phi else 0

            # in-flows from other pairs
            for v in self.network.G.nodes(data=False):
                if v != i and v != j:
                    out_pair1 = (i, v) if (i, v) in pairs else (v, i)
                    out_pair2 = (v, j) if (v, j) in pairs else (j, v)
                    self.I[in_pair] += swap_prob * 0.5 * (self.f[(out_pair1, in_pair)] + self.f[(out_pair2, in_pair)])

        # equal contribution to swap
        for in_pair in pairs:
            i, j = in_pair
            for v in nodes:
                if v != i and v != j:
                    out_pair1 = (i, v) if (i, v) in pairs else (v, i)
                    out_pair2 = (v, j) if (v, j) in pairs else (j, v)
                    self.model.addConstr(
                        self.f[(out_pair1, in_pair)] == self.f[(out_pair2, in_pair)]
                    )
        
        # available flow constraint
        for pair in pairs:
            self.model.addConstr(self.I[pair] >= self.O[pair])

        # flow conservation
        self.flow_conserv = {}
        for pair in pairs:
            self.flow_conserv[pair] = self.I[pair] - self.O[pair]

        # add flow constraints
        for pair, demand in self.network.D.items():
            self.model.addConstr(self.flow_conserv[pair] >= demand)

    def add_resource_constr(self):
        nodes = self.network.G.nodes(data=False)
        edges = self.network.G.edges(data=False)
        # memory usage constraint
        m = {node: 0 for node in nodes}
        for edge in edges:
            i, j = edge
            m[i] += self.phi[edge]
            m[j] += self.phi[edge]

        for node in nodes:
            self.model.addConstr(
                self.m[node] >= m[node]
            )

        for u, v, cap in self.network.G.edges(data='channel_capacity'):
            edge = (u, v) if (u, v) in edges else (v, u)
            self.model.addConstr(
                self.phi[edge] <= self.c[edge] * cap
            )

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
            self.pv[node] = pm * self.m[node]
            if not self.relaxed:
                self.pv[node] += pm_install * self.Im[node]
        # edge budget
        edges = self.network.G.edges(data=False)
        self.pe = {}
        for edge in edges:
            self.pe[edge] = pc * self.c[edge]
            if not self.relaxed:
                self.pe[edge] += pc_install * self.Ic[edge]

        self.budget = gp.quicksum(self.pv.values()) + gp.quicksum(self.pe.values())

    def solve(self):
        """
        optimize the model
        """
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
    task = Task(vset, 0.2, (10, 101))
    net = Network(task=task)

    net.cluster_by_nearest(5)
    # net.cluster_by_distance(500)
    net.nearest_components()
    net.plot(None, None, './result/fig_cmp.png')

    # net.make_clique_among_components()
    # net.plot(None, None, './result/fig_clique.png')
    net.segment_edge(150, 150)
    net.plot(None, None, './result/fig_segment.png')

    print(len(net.G.nodes), len(net.G.edges))
    # net.plot(None, None, './result/fig.png')

    # solver = LinearSolver(net, relaxed=True)
    solver = FlowSolver(net, relaxed=False, output=True)

    solver.solve()
    
    print("Objective value: ", solver.obj_val)
    plot_optimized_network(
        solver.network.G, 
        solver.m, solver.c, solver.phi,
        filename='./result/fig-solved.png'
        )



