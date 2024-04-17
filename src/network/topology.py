

import os
import enum

import networkx as nx


class Topology(enum.Enum):
    TEST = 'topology/test.graphml'
    ATT = 'topology/ATT.graphml'
    GETNET = 'topology/Getnet.graphml'
    # IOWA = 'topology/IowaStatewideFiberMap.graphml'
    IRIS = 'topology/Iris.graphml'
    NOEL = 'topology/Noel.graphml'


class VertexSet():

    def __init__(self, 
            topology: Topology = Topology.ATT,
            ) -> None:
        super().__init__()

        self.topology = topology

        # vertices with latitude and longitude
        # dict[str, Tuple[float, float]], {id: (latitude, longitude)}
        self.V = {}
        # read the ground network from the graphml file
        filename = topology.value
        path = os.path.join(os.path.dirname(__file__), filename)
        self.G: nx.MultiGraph = nx.read_graphml(path, force_multigraph=True)
        # extract
        for node in self.G.nodes(data=True):
            id = node[0]
            latitude, longitude = node[1]['Latitude'], node[1]['Longitude']
            self.V[id] = (latitude, longitude)





if __name__ == '__main__':
    # net = Network()
    # print(net.G.nodes)
    # print(net.G.edges(data='length', keys=True))

    vset = VertexSet()
    print(vset.V)