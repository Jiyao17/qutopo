

import os
import enum

import networkx as nx
import numpy as np


class VertexSource(enum.Enum):
    ATT = 'topology/ATT.graphml'
    GETNET = 'topology/Getnet.graphml'
    # IOWA = 'topology/IowaStatewideFiberMap.graphml'
    IRIS = 'topology/Iris.graphml'

    # CNET = 'topology/NSFCNET.graphml'
    # INS = 'topology/INS.graphml'
    # ION = 'topology/ION.graphml'
    MISSOURI = 'topology/Missouri.graphml'
    NOEL = 'topology/Noel.graphml'

    TRIANGLE = 'topology/Triangle.graphml'
    PAIR = 'topology/Pair.graphml'


class VertexSet():

    def __init__(self, 
            vsrc: VertexSource = VertexSource.ATT,
            ) -> None:
        super().__init__()

        self.name = vsrc.name

        # vertices with latitude and longitude
        # dict[str, Tuple[float, float]], {id: (latitude, longitude)}
        self.vertices = {}
        # self.edges = {}
        if isinstance(vsrc, VertexSet):
            # read the ground network from the graphml file
            filename = vsrc.value
            path = os.path.join(os.path.dirname(__file__), filename)
            self.graph: nx.MultiGraph = nx.read_graphml(path, force_multigraph=True)
            # extract
            node_id = 0
            deleted = []
            for node, data in self.graph.nodes(data=True):
                if 'Latitude' not in data or 'Longitude' not in data:
                    deleted.append(node)
                    continue
                lat, lon= data['Latitude'], data['Longitude']
                self.vertices[node_id] = (lat, lon)

                node_id += 1
        
        elif isinstance(vsrc, VertexSetRandom):
            self.vertices = vsrc.vertices

        # for node in deleted:
        #     self.graph.remove_node(node)

        # for u, v, l in self.graph.edges(data='Length', keys=False):
        #     u, v = int(u), int(v)
        #     self.edges[(u, v)] = l


class VertexSetRandom():
        """
        Generate random vertices within the given range
        """
    
        def __init__(self, 
                num: int = 10,
                lat_range: tuple = (-90, 90),
                lon_range: tuple = (-180, 180),
                ) -> None:
            super().__init__()
            self.name = 'random'+str(num)
            self.num = num
            self.lat_range = lat_range
            self.lon_range = lon_range

            self.vertices = {}

            # generate random vertices
            for i in range(num):
                lat = np.random.uniform(*lat_range)
                lon = np.random.uniform(*lon_range)
                self.vertices[i] = (lat, lon)

        def scale(self, scale: tuple=(0.01, 0.01)):
            """
            Scale the vertices
            (0.01, 0.01) for state level
            """
            for i in range(self.num):
                lat, lon = self.vertices[i]
                lat = lat * scale[0]
                lon = lon * scale[1]
                self.vertices[i] = (lat, lon)



if __name__ == '__main__':
    # net = Network()
    # print(net.G.nodes)
    # print(net.G.edges(data='length', keys=True))

    vsrc = VertexSource.NOEL
    vset = VertexSet(vsrc)

    vset = VertexSetRandom(10)
    vset.scale()
    print(len(vset.vertices))