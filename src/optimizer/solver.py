
import networkx as nx
import gurobipy as gp

from ..network import Topology, Network
from .task import NetworkConstruction

class LinearSolver():
    """
    solve the linear optimization problem
    """
    def __init__(self, task: NetworkConstruction) -> None:
        self.task = task
        self.model = gp.Model("NetworkConstruction")


    def add_variables(self):
        """
        add m, c, and f variables to the model
        """    
        # add node memory variables
        self.m = {}
        # indicator variable for memory usage
        # 1 if memory is used, 0 otherwise
        self.Im = {} 
        max_mem = 1e6
        for node in self.task.V:
            self.m[node] = self.model.addVar(
                vtype=gp.GRB.INTEGER,
                name=f'm_{node}'
                )
            self.Im[node] = self.model.addVar(
                vtype=gp.GRB.BINARY,
                name=f'Im_{node}'
                )
            self.model.addConstr(self.m[node] >= 0)
            # make Im the indicator variable
            # Im = 1 if m > 0, 0 otherwise
            self.model.addConstr(self.m[node] >= self.Im[node])
            self.model.addConstr(self.m[node] <= max_mem * self.Im[node])
         
        # add edge channel variables
        self.c = {}
        self.If = {}
        max_channel = 1e6
        for edge in self.task.E(keys=True):
            self.c[edge] = self.model.addVar(
                vtype=gp.GRB.INTEGER,
                name=f'c_{edge}'
                )
            self.If[edge] = self.model.addVar(
                vtype=gp.GRB.BINARY,
                name=f'Ic_{edge}'
                )
            self.model.addConstr(self.c[edge] >= 0)
            # make Ic the indicator variable
            # Ic = 1 if c > 0, 0 otherwise
            self.model.addConstr(self.c[edge] <= max_channel * self.If[edge])
            self.model.addConstr(self.c[edge] >= self.If[edge])
        # add edge capacity variables
        self.C = {} # \phi in the paper
        for edge in self.task.E(keys=True):
            self.C[edge] = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS,
                name=f'C_{edge}'
                )
            self.model.addConstr(self.C[edge] >= 0)
            
        pairs = self.task.pairs
        self.f = {}
        # add flow variables
        for out_pair in pairs:
            for in_pair in pairs:
                if out_pair != in_pair:
                    self.f[(out_pair, in_pair)] = self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        name=f'f_{out_pair}_{in_pair}'
                        )
                    self.model.addConstr(self.f[(out_pair, in_pair)] >= 0)

    def add_distribution_constr(self):
        """
        add the constraints to the model
        """
        # definition of out-flows
        self.O = {}
        for out_pair in self.task.pairs:
            i, j = out_pair
            self.O[(i, j)] = 0
            for v in self.task.V:
                if v != i and v != j:
                    in_pair1 = (i, v) if (i, v) in self.task.pairs else (v, i)
                    in_pair2 = (v, j) if (v, j) in self.task.pairs else (j, v)
                    self.O[out_pair] += self.f[(out_pair, in_pair1)] + self.f[(out_pair, in_pair2)]

        # definition (constraint) of in-flows
        self.I = {}
        swap_prob = self.task.swap_prob
        for in_pair in self.task.pairs:
            i, j = in_pair
            self.I[in_pair] = 0
            # entanglement generation
            edge_num = self.task.G.number_of_edges(i, j)
            for key in range(edge_num):
                self.I[in_pair] += self.C[(i, j, key)] if (i, j, key) in self.C else self.C[(j, i, key)]
            
            # in-flows from other pairs
            for v in self.task.V:
                if v != i and v != j:
                    out_pair1 = (i, v) if (i, v) in self.task.pairs else (v, i)
                    out_pair2 = (v, j) if (v, j) in self.task.pairs else (j, v)
                    self.I[in_pair] += swap_prob * 0.5 * (self.f[(out_pair1, in_pair)] + self.f[(out_pair2, in_pair)])

        # equal contribution to swap
        for in_pair in self.task.pairs:
            i, j = in_pair
            for v in self.task.V:
                if v != i and v != j:
                    out_pair1 = (i, v) if (i, v) in self.task.pairs else (v, i)
                    out_pair2 = (v, j) if (v, j) in self.task.pairs else (j, v)
                    self.model.addConstr(
                        self.f[(out_pair1, in_pair)] == self.f[(out_pair2, in_pair)]
                    )
        
        # available flow constraint
        for pair in self.task.pairs:
            self.model.addConstr(self.I[pair] >= self.O[pair])

        # flow conservation
        self.flow_conserv = {}
        for pair in self.task.pairs:
            self.flow_conserv[pair] = self.I[pair] - self.O[pair]

    def add_resource_constr(self):
        # memory usage constraint
        m = {node: 0 for node in self.task.V}
        for edge in self.task.E(keys=True):
            i, j, k = edge
            m[i] += self.C[edge]
            m[j] += self.C[edge]

        for node in self.task.V:
            self.model.addConstr(
                self.m[node] >= m[node]
            )

        for edge in self.task.E(keys=True):
            cap = self.task.E[edge]['cap_per_channel']
            self.model.addConstr(
                self.C[edge] <= self.c[edge] * cap
            )

    def add_budget_def(self):
        """
        define the budgets for the resources
        """
        # memory budget
        self.pv = {}
        for node in self.task.V:
            self.pv[node] = self.task.memory_price[node] * self.m[node]
            self.pv[node] += self.task.memory_price_install[node] * self.Im[node]
        # edge budget
        self.pe = {}
        for edge in self.task.E(keys=True):
            self.pe[edge] = self.task.fiber_price_km[edge] * self.c[edge]
            self.pe[edge] += self.task.fiber_price_install[edge] * self.If[edge]

        self.budget = gp.quicksum(self.pv.values()) + gp.quicksum(self.pe.values())

    def optimize(self):
        """
        optimize the linear model
        """

        self.add_variables()
        self.add_distribution_constr()
        self.add_resource_constr()
        self.add_budget_def()

        # add flow constraints, set the objective to minimize the budget
        for pair, demand in self.task.demands.items():
            self.model.addConstr(self.flow_conserv[pair] >= demand)
        self.model.setObjective(self.budget, gp.GRB.MINIMIZE)

        self.model.optimize()


if __name__ == "__main__":
    topology = Topology.GETNET
    network = Network(topology, scale_factor=0.1)
    task = NetworkConstruction(network)
    solver = LinearSolver(task)

    solver.optimize()
    print("Objective value: ", solver.model.objVal)