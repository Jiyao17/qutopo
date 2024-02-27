
import os
import enum

from geopy.distance import geodesic
import networkx as nx

from .topology import Topology

class GroundNetOpt(enum.Enum):
    ATT = 1
    GETNET = 2
    DARPA = 3 # unavailable for now


GROUND_FILENAME = {
    GroundNetOpt.ATT: 'ground/ATT.graphml',
    GroundNetOpt.GETNET: 'ground/Getnet.graphml',
    GroundNetOpt.DARPA: 'ground/DARPA.graphml'
}



class GroundNetTopo(Topology):

    def __init__(self, 
            ground_network: GroundNetOpt = GroundNetOpt.ATT,
            node_capacity: int = 100,
            edge_capacity: int = 100
            ) -> None:
        super().__init__()

        self.ground_network = ground_network
        filename = GROUND_FILENAME[ground_network]
        path = os.path.join(os.path.dirname(__file__), filename)
        self.net: nx.MultiGraph = nx.read_graphml(path, force_multigraph=True)

        self.set_edge_length()
        self.set_node_capacity(node_capacity)
        self.set_edge_capacity(edge_capacity)

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
            ).km
            nx.set_edge_attributes(self.net, {(src, dst, key): {'length': length}})

    def set_node_capacity(self, capacity: int = 100):
        """
        set the capacity of each node in the network
        """
        
        nodes = self.net.nodes(data=False)
        values = {node: {'capacity': capacity} for node in nodes}
        nx.set_node_attributes(self.net, values)

    def set_edge_capacity(self, capacity: int = 100):
        """
        set the capacity of each edge in the network
        """
        edges = self.net.edges
        values = {(src, dst, key): {'capacity': capacity} for src, dst, key in edges}
        nx.set_edge_attributes(self.net, values)     