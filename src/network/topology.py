

import networkx as nx

class Topology():
    def __init__(self) -> None:
        self.net: nx.MultiGraph = None

    def connected_subgraphs(self):
        return list(nx.connected_components(self.net))
    