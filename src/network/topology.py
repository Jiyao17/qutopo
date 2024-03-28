

import os
import enum

from geopy.distance import geodesic
import numpy as np
import networkx as nx


class Topology(enum.Enum):
    TEST = 'topology/test.graphml'
    ATT = 'topology/ATT.graphml'
    GETNET = 'topology/Getnet.graphml'
    # IOWA = 'topology/IowaStatewideFiberMap.graphml'
    IRIS = 'topology/Iris.graphml'
    NOEL = 'topology/Noel.graphml'


class Network():

    def __init__(self, 
            topology: Topology = Topology.ATT,
            light_freq = 10 * 1e6, # 10 MHz
            prob_loss_init: float = 0.2,
            prob_loss_length: float = 0.25,
            scale_factor: float = 1
            ) -> None:
        super().__init__()

        self.topology = topology
        self.light_freq = light_freq
        self.prob_loss_init = prob_loss_init
        self.prob_loss_length = prob_loss_length
        self.scale_factor = scale_factor

        # read the ground network from the graphml file
        filename = topology.value
        path = os.path.join(os.path.dirname(__file__), filename)
        self.G: nx.MultiGraph = nx.read_graphml(path, force_multigraph=True)
        self.mark_initial_nodes()

        self.set_edge_length()
        self.scale(scale_factor)
        self.set_node_capacity(0)
        self.set_channel_capacity(prob_loss_init, prob_loss_length)

    def mark_initial_nodes(self):
        """
        mark the initial nodes in the network
        """
        marks = {}
        for node in self.G.nodes(data=False):
            marks[node] = {'initial': 1}
        nx.set_node_attributes(self.G, marks)

    def set_edge_length(self):
        """
        set the length of each edge in the network
        """
        for edge in self.G.edges(keys=True):
            src, dst, key = edge
            length = geodesic(
                (self.G.nodes[src]['Latitude'], self.G.nodes[src]['Longitude']),
                (self.G.nodes[dst]['Latitude'], self.G.nodes[dst]['Longitude'])
            ).km
            nx.set_edge_attributes(self.G, {(src, dst, key): {'length': length}})

    def set_node_capacity(self, capacity: int = 100):
        """
        set the capacity of each node in the network
        """
        nodes = self.G.nodes(data=False)
        values = {node: {'capacity': capacity} for node in nodes}
        nx.set_node_attributes(self.G, values)

    def set_channel_capacity(self, p_loss_init: float=0.2, p_loss_length: float=0.25):
        """
        set the capacity of each edge in the network
        """

        edges = self.G.edges(keys=True, data='length')
        values = {}
        for u, v, k, l in edges:
            prob_succ = (1 - p_loss_init) * np.power(10, - l * p_loss_length / 10)
            capacity = self.light_freq * prob_succ
            values[(u, v, k)] = {'cap_per_channel': capacity}
        nx.set_edge_attributes(self.G, values)

    def scale(self, scale_factor: float):
        """
        scale the edges in the network by a factor
        """
        # set new edge length
        new_length = {}
        for src, dst, key, length in self.G.edges(keys=True, data='length'):
            new_length[(src, dst, key)] = {'length': length * scale_factor}
        nx.set_edge_attributes(self.G, new_length)

        # # update edge capacity
        # self.set_channel_capacity(self.prob_loss_init, self.prob_loss_length)    


if __name__ == '__main__':
    net = Network()
    print(net.G.nodes)
    print(net.G.edges(data='length', keys=True))