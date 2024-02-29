
import networkx as nx
import gurobipy as gp
import numpy as np

from ..network import *


class QuTopoTask():

    def __init__(self,
            topo: FusedTopo,
            swap_prob: float=0.5,
            budget: float=1e6,
            demands: dict=None,
            vertex_prices: dict=None,
            edge_prices: dict=None,
        ) -> None:

        self.topo = topo
        self.swap_prob = swap_prob
        self.budget = budget
        self.pairs = self.get_pairs()
        
        if demands is None:
            demands = {pair: 1 for pair in self.pairs}
        self.demands = demands

        if vertex_prices is None:
            vertex_prices = {node: 10 for node in self.topo.net.nodes}
        self.vertex_prices = vertex_prices

        if edge_prices is None:
            edge_prices = {
                (u, v, k) : l
                for u, v, k, l in self.topo.net.edges(data='length', keys=True)
                }
        self.edge_price = edge_prices

    def get_pairs(self):
        """
        get all pairs of nodes in the network
        """

        nodes = self.topo.net.nodes
        pairs = []
        for src in nodes:
            for dst in nodes:
                if src != dst and (dst, src) not in pairs:
                    # networkx use string as node names
                    pairs.append((str(src), str(dst)))

        return pairs

        

    
