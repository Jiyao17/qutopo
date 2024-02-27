

import networkx as nx

from .topology import Topology
from .ground import GroundNetTopo
from .constellation import ConstellationPosition

class FusedTopo(Topology):
    def __init__(self, 
        ground_topology: GroundNetTopo,
        constellation: ConstellationPosition
        ) -> None:
        super().__init__()
        self.ground_topology = ground_topology
        self.constellation = constellation
        self.net = self.fuse_network()

    def fuse_network(self):
        """
        fuse the ground and constellation networks
        """
        net = self.ground_topology.net
        nx.set_node_attributes(net, values='ground', name='type')
        # for node in self.constellation.constellation:
        #     net.add_node(node, **self.constellation.constellation[node])
        return net