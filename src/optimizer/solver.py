
import networkx as nx
import gurobipy as gp

from ..network import *
from .task import QuTopoTask


class LinearSolver():
    """
    solve the linear optimization problem
    """
    def __init__(self, task: QuTopoTask, var_type=gp.GRB.CONTINUOUS) -> None:
        self.task = task
        self.var_type = var_type
        self.model = gp.Model("QuTopo")

        self.export_task()

    def add_variables(self):
        """
        add m, c, and f variables to the model
        """    
        # add node memory variables
        self.m = {}
        for node in self.task.topo.net.nodes:
            self.m[node] = self.model.addVar(
                vtype=gp.GRB.INTEGER,
                name=f'm_{node}'
                )
            self.model.addConstr(self.m[node] >= 0)
            
        # add edge capacity variables
        self.c = {}
        for edge in self.task.topo.net.edges:
            self.c[edge] = self.model.addVar(
                vtype=gp.GRB.INTEGER,
                name=f'c_{edge}'
                )
            self.model.addConstr(self.c[edge] >= 0)
            
        pairs = self.task.pairs
        self.f = {}
        # add flow variables
        for out_pair in pairs:
            for in_pair in pairs:
                if out_pair != in_pair:
                    self.f[(out_pair, in_pair)] = self.model.addVar(
                        vtype=gp.GRB.INTEGER,
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
            for v in self.task.topo.net.nodes:
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
            edge_num = self.task.topo.net.number_of_edges(i, j)
            for key in range(edge_num):
                prob = self.task.topo.net[i][j][key]['prob']
                self.I[in_pair] += self.c[(i, j, key)] * prob
            
            # in-flows from other pairs
            for v in self.task.topo.net.nodes:
                if v != i and v != j:
                    out_pair1 = (i, v) if (i, v) in self.task.pairs else (v, i)
                    out_pair2 = (v, j) if (v, j) in self.task.pairs else (j, v)
                    self.I[in_pair] += swap_prob * 0.5 * (self.f[(out_pair1, in_pair)] + self.f[(out_pair2, in_pair)])
    
        # equal contribution to swap
        for in_pair in self.task.pairs:
            i, j = in_pair
            for v in self.task.topo.net.nodes:
                if v != i and v != j:
                    out_pair1 = (i, v) if (i, v) in self.task.pairs else (v, i)
                    out_pair2 = (v, j) if (v, j) in self.task.pairs else (j, v)
                    self.model.addConstr(
                        self.f[(out_pair1, in_pair)] == self.f[(out_pair2, in_pair)]
                    )
                    self.model.addConstr(self.f[(out_pair1, in_pair)] >= 0)
                    self.model.addConstr(self.f[(out_pair2, in_pair)] >= 0)
        
        # available flow constraint
        for pair in self.task.pairs:
            self.model.addConstr(self.I[pair] >= self.O[pair])

        # demand constraint
        for pair in self.task.demands.keys():
            self.model.addConstr(
                self.I[pair] - self.O[pair] >= self.task.demands[pair]
            )

    def add_resource_constr(self, channel_mem: bool=True):
        m = {node: 0 for node in self.task.topo.net.nodes}
        for edge in self.task.topo.net.edges:
            i, j, k = edge
            if channel_mem:
                # one memory cell per channel
                prob = 1
            else:
                # shared memory among channels
                prob = self.task.topo.net[i][j][k]['prob']
            m[i] += self.c[edge] * prob
            m[j] += self.c[edge] * prob

        for node in self.task.topo.net.nodes:
            self.model.addConstr(
                self.m[node] >= m[node]
            )

    def add_budget_constr(self):
        """
        define the budgets for the resources
        """
        # vertex budget
        self.pv = {node: 0 for node in self.task.topo.net.nodes}
        for node in self.task.topo.net.nodes:
            self.pv[node] += self.task.vertex_prices[node] * self.m[node]
        # edge budget
        self.pe = {edge: 0 for edge in self.task.topo.net.edges}
        for edge in self.task.topo.net.edges:
            self.pe[edge] += self.task.edge_price[edge] * self.c[edge]

    def export_task(self):
        """
        export the graph parameters to linear solver gurobi
        """
        self.add_variables()
        self.add_distribution_constr()
        self.add_resource_constr()
        self.add_budget_constr()

    def optimize(self):
        """
        optimize the linear model
        """
        objective = gp.quicksum(self.pv.values()) + gp.quicksum(self.pe.values())
        self.model.setObjective(objective, gp.GRB.MINIMIZE)
        self.model.optimize()


if __name__ == "__main__":
    att = GroundNetTopo(GroundNetOpt.GETNET)
    gps = ConstellationPosition(ConstellationOpt.GPS)
    topo = FusedTopo(att, gps)
    task = QuTopoTask(topo, 1)
    solver = LinearSolver(task)
    solver.optimize()
    print(solver.model.objVal)