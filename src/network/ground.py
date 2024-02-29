
import os
import enum

from geopy.distance import geodesic
import networkx as nx
import netsquid as ns
import numpy as np

from .topology import Topology

class GroundNetOpt(enum.Enum):
    ATT = 1
    GETNET = 2
    IOWA = 3
    IRIS = 4
    NOEL = 5


GROUND_FILENAME = {
    GroundNetOpt.ATT: 'ground/ATT.graphml',
    GroundNetOpt.GETNET: 'ground/Getnet.graphml',
    GroundNetOpt.IOWA: 'ground/IowaStatewideFiberMap.graphml',
    GroundNetOpt.IRIS: 'ground/Iris.graphml',
    GroundNetOpt.NOEL: 'ground/Noel.graphml',
}



class GroundNetTopo(Topology):

    def __init__(self, 
            ground_network: GroundNetOpt = GroundNetOpt.IOWA,
            node_capacity: int = 100,
            edge_base_capacity: int = 100,
            prob_loss_init: float = 0.2,
            prob_loss_length: float = 0.25,
            ) -> None:
        super().__init__()

        self.ground_network = ground_network
        self.node_capacity = node_capacity
        self.edge_base_capacity = edge_base_capacity
        self.prob_loss_init = prob_loss_init
        self.prob_loss_length = prob_loss_length

        # read the ground network from the graphml file
        filename = GROUND_FILENAME[ground_network]
        path = os.path.join(os.path.dirname(__file__), filename)
        self.net: nx.MultiGraph = nx.read_graphml(path, force_multigraph=True)

        self.set_edge_length()
        self.set_node_capacity(node_capacity)
        self.set_edge_prob(prob_loss_init, prob_loss_length)

    def set_edge_length(self):
        """
        set the length of each edge in the network
        """
        for edge in self.net.edges(data=True):
            src, dst = edge[:2]
            key = edge[2]['key']
            length = geodesic(
                (self.net.nodes[src]['Latitude'], self.net.nodes[src]['Longitude']),
                (self.net.nodes[dst]['Latitude'], self.net.nodes[dst]['Longitude'])
            ).km / 100
            nx.set_edge_attributes(self.net, {(src, dst, key): {'length': length}})

    def set_node_capacity(self, capacity: int = 100):
        """
        set the capacity of each node in the network
        """
        nodes = self.net.nodes(data=False)
        values = {node: {'capacity': capacity} for node in nodes}
        nx.set_node_attributes(self.net, values)

    def set_edge_prob(self, p_loss_init: float=0.2, p_loss_length: float=0.25):
        """
        set the capacity of each edge in the network
        """

        edges = self.net.edges(data=True, keys=True)
        values = {}
        for u, v, k, d in edges:
            length = d['length']
            prob_succ = (1 - p_loss_init) * np.power(10, - length * p_loss_length / 10)
            values[(u, v, k)] = {'prob': prob_succ}
        nx.set_edge_attributes(self.net, values)


if __name__ == '__main__':
    grd = GroundNetTopo(GroundNetOpt.GETNET)
    prob = {}

    print(prob)