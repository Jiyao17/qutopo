
import numpy as np
import networkx as nx
import geopy.distance as geo

from .task import Task
from ..utils.plot import plot_nx_graph




class Network:

    def __init__(self, task: Task = Task()) -> None:
        self.task = task
        self.V = task.V
        self.D = task.D
        self.pairs = task.pairs

        self.area = {
            'lat_max': np.max([v[0] for v in self.V.values()]),
            'lat_min': np.min([v[0] for v in self.V.values()]),
            'lon_max': np.max([v[1] for v in self.V.values()]),
            'lon_min': np.min([v[1] for v in self.V.values()])
            }
        
        self.G: 'nx.Graph' = nx.Graph()
        self.G.add_nodes_from(self.V, original=True)

    def get_random_point(self, area: dict):
        """
        get a random point within the area
        """
        lat = np.random.uniform(area['lat_min'], area['lat_max'])
        lon = np.random.uniform(area['lon_min'], area['lon_max'])
        return lat, lon

    def make_clique(self):
        """
        make a clique network
        """
        # add edges if not exist
        for node in self.V:
            for neighbor in self.V:
                if node != neighbor and not self.G.has_edge(node, neighbor):
                    self.G.add_edge(node, neighbor)
        self.update_edge_length()

    def update_edge_length(self):
        """
        update the edge length based on the distance between nodes
        """
        for edge in self.G.edges:
            u, v = edge
            length = geo.distance(self.V[u], self.V[v]).km
            # round to 2 decimal places
            self.G[u][v]['length'] = np.round(length, 2)

    def prune_edge_by_length(self, threshold: float=100):
        """
        remove edges with length greater than threshold
        """
        for edge in self.G.edges:
            if self.G[edge[0]][edge[1]]['length'] > threshold:
                self.G.remove_edge(*edge)
    
    def prune_edge_by_density(self, lb:5):
        """
        remove edges with density less than lb
        """
        pass
    
    def plot(self, labeled_edges: bool=False, filename: str='./result/fig.png'):
        plot_nx_graph(self.G, labeled_edges, filename)



if __name__ == '__main__':
    net = Network()
    net.make_clique()

    print(net.G.nodes(data='original'))
    print(net.G.edges(data='length'))
    
    net.plot()

    # net.prune_edge_by_length(500)
    # net.plot()
