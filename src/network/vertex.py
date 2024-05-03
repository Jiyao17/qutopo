

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
        # self.edges = {}
        # read the ground network from the graphml file
        filename = vsrc.value
        path = os.path.join(os.path.dirname(__file__), filename)
        self.graph: nx.MultiGraph = nx.read_graphml(path, force_multigraph=True)
        # extract
        for node, data in self.graph.nodes(data=True):
            id = int(node)
            lat, lon= data['Latitude'], data['Longitude']
            self.vertices[id] = (lat, lon)

        # for u, v, l in self.graph.edges(data='Length', keys=False):
        #     u, v = int(u), int(v)
        #     self.edges[(u, v)] = l





if __name__ == '__main__':
    # net = Network()
    # print(net.G.nodes)
    # print(net.G.edges(data='length', keys=True))

    vset = VertexSet()
    print(vset.vertices)