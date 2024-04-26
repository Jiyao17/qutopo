
import numpy as np
import networkx as nx
import geopy.distance as geo

from .task import Task
from .vertex import VertexSet, VertexSource
from .quantum import complete_swap
from ..utils.plot import plot_nx_graph


HWParam = {
    'swap_prob': 1, # swap probability
    'fiber_loss': 0.2, # fiber loss
    'photon_rate': 1e4, # photon rate
    'pm': 1e1, # memory price per slot
    # 'pm_install': 1e6, # memory installation price
    'pm_install': 1e2, # memory installation price
    'pc': 1, # channel price per km
    # 'pc_install': 1e4, # channel installation price
    'pc_install': 0, # channel installation price
}


class Network:

    def __init__(self, 
            task: Task = Task(),
            hw_params: dict = HWParam
            ) -> None:
        self.task = task
        self.hw_params = hw_params

        self.U = task.U # only original nodes
        self.D = task.D # demands

        # square area enclosing all nodes, 
        self.area = {
            'lon_min': min([v[0] for v in self.U.values()]),
            'lon_max': max([v[0] for v in self.U.values()]),
            'lat_min': min([v[1] for v in self.U.values()]),
            'lat_max': max([v[1] for v in self.U.values()])
            }
        
        self.G: 'nx.Graph' = nx.Graph()
        for node in self.U.keys():
            node_id = node
            lon, lat = self.U[node]
            self.G.add_node(node_id, pos=(lon, lat), original=True)

        self.pairs = []

    def connect_nearest_nodes(self, num: int=3):
        """
        cluster nearby nodes
        """
        # connect nearest num nodes
        
        for u in self.G.nodes(data=False):
            nearest_found = []
            for v in self.G.nodes(data=False):
                if u != v :
                    pos1 = self.G.nodes[u]['pos']
                    pos2 = self.G.nodes[v]['pos']
                    distance = geo.distance((pos1[1], pos1[0]), (pos2[1], pos2[0])).km
                    if len(nearest_found) < num:
                        nearest_found.append((v, distance))
                    else:
                        nearest_found.sort(key=lambda x: x[1])
                        if distance < nearest_found[-1][1]:
                            nearest_found[-1] = (v, distance)
            for v, _ in nearest_found:
                self.G.add_edge(u, v, length=distance)

        self.update_edges()

    def connect_nearest_component(self):
        """
        connect the graph by repeatedly connecting the nearest components
        """
        components = list(nx.connected_components(self.G))
        while len(components) > 1:
            min_dist = np.inf
            min_edge = None
            for i in range(len(components) - 1):
                for j in range(i + 1, len(components)):
                    # find the nearest pair of nodes
                    for u in components[i]:
                        for v in components[j]:
                            pos1 = self.G.nodes[u]['pos']
                            pos2 = self.G.nodes[v]['pos']
                            distance = geo.distance((pos1[1], pos1[0]), (pos2[1], pos2[0])).km
                            if distance < min_dist:
                                min_dist = distance
                                min_edge = (u, v)
            self.G.add_edge(*min_edge, length=min_dist)
            components = list(nx.connected_components(self.G))

        self.update_edges()

    def segment_edge(self, threshold: float, seg_len: float):
        """
        segment the edge with length greater than threshold
        """
        nodes_to_add = []
        edges_to_add = []
        edges_to_remove = []
        new_id = len(self.G.nodes)
        for u, v, l in self.G.edges(data='length'):
            if l > threshold:
                edges_to_remove.append((u, v))
                u_lon, u_lat = self.G.nodes[u]['pos']
                v_lon, v_lat = self.G.nodes[v]['pos']
                point_num = int(np.ceil(l / seg_len)) - 1
                # add seg_num nodes between u and v
                for i in range(1, point_num + 1):
                    lon = u_lon + (v_lon - u_lon) / (point_num + 1) * i
                    lat = u_lat + (v_lat - u_lat) / (point_num + 1) * i
                    nodes_to_add.append((str(new_id), (lon, lat)))

                    if i == 1:
                        edges_to_add.append((u, str(new_id)))
                    elif i <= point_num:
                        # nodes_to_add[-1] is the current node
                        edges_to_add.append((str(new_id), nodes_to_add[-2][0]))
                    if i == point_num:
                        edges_to_add.append((str(new_id), v))

                    new_id += 1

        for node_id, pos in nodes_to_add:
            self.G.add_node(node_id, pos=pos, original=False)
        for edge in edges_to_add:
            self.G.add_edge(*edge)
        for edge in edges_to_remove:
            self.G.remove_edge(*edge)

        self.update_edges()
        self.update_pairs()

    def cluster_inter_nodes(self, k: int=3):
        """
        merge close intermediate nodes
        1. cluster the intermediate nodes by k-means
        2. merge the nodes in the same cluster
        """
        
        nodes = [node for node in self.G.nodes(data=False) if not self.G.nodes[node]['original']]
        if k > len(nodes):
            k = len(nodes)
        # pick k random nodes as initial cluster centers
        initial_center_nodes = np.random.choice(nodes, k, replace=False)
        centers = np.array([self.G.nodes[node]['pos'] for node in initial_center_nodes])
        clusters: 'list[list]'= [ [ ] for _ in range(k)]
        new_centers = np.zeros((k, 2))
        # k-means
        while True:
            for node in nodes:
                pos = self.G.nodes[node]['pos']
                min_dist = np.inf
                min_center = 0
                for i, center in enumerate(centers):
                    dist = geo.distance((center[1], center[0]), (pos[1], pos[0])).km
                    if dist < min_dist:
                        min_dist = dist
                        min_center = i
                clusters[min_center].append(node)
            for i, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    new_centers[i] = centers[i]
                else:
                    new_centers[i] = np.mean([self.G.nodes[node]['pos'] for node in cluster], axis=0)
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
            clusters = [ [ ] for _ in range(k)]
            
        # remove all intermediate nodes
        for node in nodes:
            self.G.remove_node(node)
        # add new nodes
        new_id = len(self.G.nodes)
        for i, cluster in enumerate(clusters):
            lon, lat = centers[i]
            self.G.add_node(str(new_id), pos=(lon, lat), original=False)
            new_id += 1

        self.update_edges()
        self.update_pairs()



    def update_pairs(self):
        """
        get all pairs of nodes in the network
        """
        self.pairs = []
        nodes = list(self.G.nodes(data=False))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                self.pairs.append((nodes[i], nodes[j]))
        return self.pairs

    def update_edges(self):
        """
        update the edge length and channel_capacity
        based on the distance between nodes
        """
        for edge in self.G.edges(data=False):
            u, v = edge
            u_pos = self.G.nodes[u]['pos']
            v_pos = self.G.nodes[v]['pos']
            length = geo.distance((u_pos[1], u_pos[0]), (v_pos[1], v_pos[0])).km
            fiber_loss = self.hw_params['fiber_loss']
            photon_rate = self.hw_params['photon_rate']
            # prob for half fiber (detectors are in the middle of edges)
            prob = 10 ** (-0.1 * fiber_loss * (length/2)) 
            channel_capacity = photon_rate * prob**2
            # round to 2 decimal places
            self.G[u][v]['length'] = np.round(length, 2)
            self.G[u][v]['channel_capacity'] = channel_capacity

    def plot(self, 
            node_label: str='id', 
            edge_label: str='length', 
            filename: str='./result/fig.png'
            ):
        plot_nx_graph(self.G, node_label, edge_label, filename)




if __name__ == '__main__':
    topo = VertexSource.NOEL
    vset = VertexSet(topo)
    task = Task(vset)
    net = Network(task)

    # net.cluster(300)
    # net.plot(None, None, './result/fig_cluster.png')

    # net.add_grid_points(width=500)
    # net.plot(None, None, './result/fig_grid.png')

    city_num = len(vset.vertices)

    net.connect_nearest_nodes(num=5)
    net.plot(None, None, './result/test/fig_cluster.png')

    net.connect_nearest_component()
    net.plot(None, None, './result/test/fig_nearest.png')

    net.segment_edge(100, 100)
    net.plot(None, None, './result/test/fig_segment.png')

    inter_node_num = len(net.G.nodes) - city_num

    net.cluster_inter_nodes(inter_node_num // 3)
    net.plot(None, None, './result/test/fig_merge.png')

    net.connect_nearest_nodes(num=5)
    net.plot(None, None, './result/test/fig_cluster2.png')

    net.connect_nearest_component()
    net.plot(None, None, './result/test/fig_nearest2.png')

    net.segment_edge(100, 100)
    net.plot(None, None, './result/test/fig_segment2.png')






    # print(net.G.nodes(data=True))
    # print(net.G.edges(data=True))
    # print(len(net.G.nodes), len(net.G.edges))


    
