
import networkx as nx
import gurobipy as gp
import numpy as np

from .vertex import VertexSet, VertexSetRandom


class Task():

    def __init__(self,
            vset: VertexSet = VertexSet(),
            demand_frac: float = 0.2,
            demand_range: tuple = (10, 11),
        ) -> None:

        self.demand_frac = demand_frac
        self.demand_range = demand_range

        self.U = vset.vertices
        self.D = {}
        
        self.pairs = self.get_pairs(self.U.keys())
        dpair_num = int(len(self.pairs) * demand_frac)
        indices = np.arange(len(self.pairs))
        dpairs_indices = np.random.choice(indices, dpair_num, replace=False)
        dpairs = [self.pairs[i] for i in dpairs_indices]
        for pair in dpairs:
            self.D[pair] = np.random.randint(demand_range[0], demand_range[1])
        for pair in self.pairs:
            if pair not in self.D:
                self.D[pair] = 0
          
    def get_pairs(self, nodes: list):
        """
        get all pairs of nodes in the network
        """
        pairs: 'list[tuple[str, str]]' = []
        for src in nodes:
            for dst in nodes:
                if src != dst and (dst, src) not in pairs:
                    pairs.append((src, dst))

        return pairs



if __name__ == '__main__':

    vset = VertexSetRandom(num=20)
    vset.scale((0.01, 0.01))

    task = Task(vset, 0.5, (10, 11))
    print(task.U)
    print(task.D)