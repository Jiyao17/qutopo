
import networkx as nx
import gurobipy as gp

from ..network import *
from .task import _QuTopoTask, QuTopoConstruction


class LinearSolver():
    """
    solve the linear optimization problem
    """
    def __init__(self, task: _QuTopoTask, var_type=gp.GRB.INTEGER) -> None:
        self.task = task
        self.var_type = var_type
        self.model = gp.Model("QuTopo")


    def add_variables(self, var_type=gp.GRB.INTEGER):
        """
        add m, c, and f variables to the model
        """    
        # add node memory variables
        self.m = {}
        for node in self.task.V:
            self.m[node] = self.model.addVar(
                vtype=var_type,
                name=f'm_{node}'
                )
            self.model.addConstr(self.m[node] >= 0)
            
        # add edge channel variables
        self.c = {}
        for edge in self.task.E:
            self.c[edge] = self.model.addVar(
                vtype=var_type,
                name=f'c_{edge}'
                )
            self.model.addConstr(self.c[edge] >= 0)
        # add edge capacity variables
        self.C = {}
        for edge in self.task.E:
            self.C[edge] = self.model.addVar(
                vtype=var_type,
                name=f'C_{edge}'
                )
            
        pairs = self.task.pairs
        self.f = {}
        # add flow variables
        for out_pair in pairs:
            for in_pair in pairs:
                if out_pair != in_pair:
                    self.f[(out_pair, in_pair)] = self.model.addVar(
                        vtype=var_type,
                        name=f'f_{out_pair}_{in_pair}'
                        )
                    self.model.addConstr(self.f[(out_pair, in_pair)] >= 0)

    def add_distribution_constr(self):
        """
        add the constraints to the model
        """
        # out-flows
        self.O = {}
        for out_pair in self.task.pairs:
            i, j = out_pair
            self.O[(i, j)] = 0
            for v in self.task.V:
                if v != i and v != j:
                    in_pair1 = (i, v) if (i, v) in self.task.pairs else (v, i)
                    in_pair2 = (v, j) if (v, j) in self.task.pairs else (j, v)
                    self.O[out_pair] += self.f[(out_pair, in_pair1)] + self.f[(out_pair, in_pair2)]

        # in-flows
        self.I = {}
        swap_prob = self.task.swap_prob
        for in_pair in self.task.pairs:
            i, j = in_pair
            self.I[in_pair] = 0
            # entanglement generation
            edge_num = self.task.G.number_of_edges(i, j)
            for key in range(edge_num):
                self.I[in_pair] += self.C[(i, j, key)] 
            
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
        m = {node: 0 for node in self.task.V}
        for edge in self.task.E:
            i, j, k = edge
            m[i] += self.C[edge]
            m[j] += self.C[edge]

        for node in self.task.V:
            self.model.addConstr(
                self.m[node] >= m[node]
            )

        for edge in self.task.V:
            i, j, k = edge
            capacity = self.task.G[i][j][k]['cap_per_channel']
            self.model.addConstr(
                self.C[edge] <= self.c[edge] * capacity
            )

    def add_budget_def(self):
        """
        define the budgets for the resources
        """
        # vertex budget
        self.pv = {}
        for node in self.task.V:
            self.pv[node] = self.task.vertex_prices[node] * self.m[node]
        # edge budget
        self.pe = {}
        for edge in self.task.g0.net.edges:
            self.pe[edge] = self.task.edge_price[edge] * self.c[edge]

        self.budget = gp.quicksum(self.pv.values()) + gp.quicksum(self.pe.values())

    def optimize(self, obj: str='budget'):
        """
        optimize the linear model
            -obj: the objective function to optimize
                budget: minimize the budget
                flow: maximize the flow
        """

        self.add_variables()
        self.add_distribution_constr()
        self.add_resource_constr()
        self.add_budget_def()

        if obj == 'budget':
            # add flow constraints, set the objective to minimize the budget
            for pair, demand in self.task.demands.items():
                self.model.addConstr(self.flow_conserv[pair] >= demand)
            self.model.setObjective(self.budget, gp.GRB.MINIMIZE)
        elif obj == 'flow':
            # add budget constraints, set the objective to maximize the flow
            self.model.addConstr(self.budget <= self.task.budget)
            flow = gp.quicksum([self.flow_conserv[pair] for pair in self.task.demands])
            self.model.setObjective(flow, gp.GRB.MAXIMIZE)

        self.model.optimize()


if __name__ == "__main__":
    att = GroundNetTopo(GroundNetOpt.GETNET)
    gps = ConstellationPosition(ConstellationOpt.GPS)
    topo = FusedTopo(att, gps)
    task = QuTopoConstruction(topo, 1)
    solver = LinearSolver(task)
    solver.optimize()
    print(solver.model.objVal)