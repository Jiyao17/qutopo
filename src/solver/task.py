
import networkx as nx
import gurobipy as gp
import numpy as np


from ..network import Topology, Task

class NetworkConstruction(_Task):
    def __init__(self, 
            network: Network,
            swap_prob: float = 0.5,
            fiber_price_km: float = 1,
            fiber_price_install: float = 1e3,
            memory_price: float = 1,
            memory_price_install: float = 1e3,
            ) -> None:
        super().__init__(network, swap_prob)

        self.demands = { pair: 10 for pair in self.pairs }
        self.memory_price = { node: memory_price for node in self.V }
        self.memory_price_install = { node: memory_price_install for node in self.V }
        self.fiber_price_km = {
            (u, v, k) : l * fiber_price_km
                for u, v, k, l in self.E(data='length', keys=True)
        }
        self.fiber_price_install = {
            (u, v, k) : fiber_price_install
                for u, v, k in self.E(keys=True)
        }


if __name__ == '__main__':
    # net = Network()
    # task = NetworkConstruction(net)
    # print(task.V)

    vset = VertexSet()
    task = Task(vset)
    print(task.V)
    print(task.D)