
import networkx as nx
import gurobipy as gp
import numpy as np

from ..network import Network


class _Task():

    def __init__(self,
            network: Network,
            swap_prob: float=0.5,
        ) -> None:

        self.network = network
        self.swap_prob = swap_prob

        self.G = network.G
        self.V = network.G.nodes
        self.E = network.G.edges

        self.pairs = self.get_pairs(list(set(self.V)))
        
    def get_pairs(self, nodes: list):
        """
        get all pairs of nodes in the network
        """
        pairs: 'list[tuple[str, str]]' = []
        for src in nodes:
            for dst in nodes:
                if src != dst and (dst, src) not in pairs:
                    # networkx use string as node names
                    pairs.append((str(src), str(dst)))

        return pairs


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
    net = Network()
    task = NetworkConstruction(net)
    print(task.V)