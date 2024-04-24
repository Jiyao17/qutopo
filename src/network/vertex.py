

import os
import enum

import networkx as nx


class VertexSource(enum.Enum):
    ATT = 'topology/ATT.graphml'
    GETNET = 'topology/Getnet.graphml'
    # IOWA = 'topology/IowaStatewideFiberMap.graphml'
    IRIS = 'topology/Iris.graphml'
    NOEL = 'topology/Noel.graphml'
    TRIANGLE = 'topology/Triangle.graphml'
    PAIR = 'topology/TwoNodes.graphml'


class VertexSet():

    def __init__(self, 
            vsrc: VertexSource = VertexSource.ATT,
            ) -> None:
        super().__init__()

        self.vsrc = vsrc

        # vertices with latitude and longitude
        # dict[str, Tuple[float, float]], {id: (latitude, longitude)}
        self.vertices = {}
        # read the ground network from the graphml file
        filename = vsrc.value
        path = os.path.join(os.path.dirname(__file__), filename)
        self.G: nx.MultiGraph = nx.read_graphml(path, force_multigraph=True)
        # extract
        for node in self.G.nodes(data=True):
            id = node[0]
            lon, lat = node[1]['Longitude'], node[1]['Latitude']
            self.vertices[id] = (lon, lat)





if __name__ == '__main__':
    # net = Network()
    # print(net.G.nodes)
    # print(net.G.edges(data='length', keys=True))

    vset = VertexSet()
    print(vset.vertices)