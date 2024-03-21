
import networkx as nx
import gurobipy as gp
import numpy as np

from src.network import FusedTopo

from ..network import *


class _QuTopoTask():

    def __init__(self,
            topo: FusedTopo,
            swap_prob: float=0.5,
        ) -> None:

        self.topo = topo
        self.swap_prob = swap_prob

        self.G = topo.net
        self.V = topo.net.nodes
        self.E = topo.net.edges

        self.pairs = self.get_pairs(list(set(self.V)))
        


    def get_pairs(self, nodes: list):
        """
        get all pairs of nodes in the network
        """
        pairs = []
        for src in nodes:
            for dst in nodes:
                if src != dst and (dst, src) not in pairs:
                    # networkx use string as node names
                    pairs.append((str(src), str(dst)))

        return pairs


class QuTopoConstruction(_QuTopoTask):
    def __init__(self, 
            topo: FusedTopo,
            swap_prob: float = 0.5,
            fiber_price: float = 1,
            memory_price: float = 1,
            ) -> None:
        super().__init__(topo, swap_prob)

        self.demands = { pair: 1 for pair in self.pairs }
        self.memory_price = { node: memory_price for node in self.V }
        self.edge_price = { 
            (u, v, k) : l * fiber_price
                for u, v, k, l in self.E(data='length', keys=True)
        }

class QuTopoRental(_QuTopoTask):
    def __init__(self, 
            topo: FusedTopo,
            swap_prob: float = 0.5,
            fiber_price: dict = None,
            memory_price: dict = None,
            ) -> None:
        super().__init__(topo, swap_prob)

        self.demands = { pair: 1 for pair in self.pairs }
        self.memory_price = { node: memory_price for node in self.V }
        self.edge_price = { 
            (u, v, k) : l * fiber_price
                for u, v, k, l in self.E(data='length', keys=True)
        }


class QuTopoEnhancement(_QuTopoTask):
    pass
