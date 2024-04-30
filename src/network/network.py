
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
    'pm': 1, # memory price per slot
    # 'pm_install': 1e6, # memory installation price
    'pm_install': 1e1, # memory installation price
    'pc': 1, # channel price per km
    # 'pc_install': 1e4, # channel installation price
    'pc_install': 1e1, # channel installation price
}


class Topology:

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
            'lat_min': min([v[1] for v in self.U.values()]),
            'lat_max': max([v[1] for v in self.U.values()]),
            'lon_min': min([v[0] for v in self.U.values()]),
            'lon_max': max([v[0] for v in self.U.values()]),
            }
        
        self.G: 'nx.Graph' = nx.Graph()
        for node in self.U.keys():
            node_id = node
            pos = self.U[node]
            self.G.add_node(node_id, pos=pos, group=0)

        self.pairs = self.update_pairs()

    def connect_nearest_nodes(self, num: int=5, group: int=0):
        """
        cluster nearby nodes
        """
        # connect nearest num nodes
        
        for u in self.G.nodes(data=False):
            nearest_found = []
            for v in self.G.nodes(data=False):
                if u != v :
                    u_pos = self.G.nodes[u]['pos']
                    v_pos = self.G.nodes[v]['pos']
                    distance = geo.distance(u_pos, v_pos).km
                    if len(nearest_found) < num:
                        nearest_found.append((v, distance))
                    else:
                        nearest_found.sort(key=lambda x: x[1])
                        if distance < nearest_found[-1][1]:
                            nearest_found[-1] = (v, distance)
            for v, _ in nearest_found:
                self.G.add_edge(u, v, length=distance, group=group)

        self.update_edges()

    def connect_nearest_component(self, group: int=0):
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
                            u_pos = self.G.nodes[u]['pos']
                            v_pos = self.G.nodes[v]['pos']
                            distance = geo.distance(u_pos, v_pos).km
                            if distance < min_dist:
                                min_dist = distance
                                min_edge = (u, v)
            self.G.add_edge(*min_edge, length=min_dist, group=group)
            components = list(nx.connected_components(self.G))

        self.update_edges()

    def make_clique(self, nodes: list, group: int=0):
        """
        form a clique for the nodes
        """
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u = nodes[i]
                v = nodes[j]
                u_pos = self.G.nodes[u]['pos']
                v_pos = self.G.nodes[v]['pos']
                distance = geo.distance(u_pos, v_pos).km
                self.G.add_edge(u, v, length=distance, group=group)

    def segment_edge(self, u, v, point_num, group=0):
        """
        segment the edge with length greater than threshold
        form clique for the whole edge
        """
        u_lat, u_lon = self.G.nodes[u]['pos']
        v_lat, v_lon = self.G.nodes[v]['pos']
        new_id = len(self.G.nodes)
        clique_nodes = [u, v]
        # add seg_num nodes between u and v
        for i in range(1, point_num + 1):
            lat = u_lat + (v_lat - u_lat) / (point_num + 1) * i
            lon = u_lon + (v_lon - u_lon) / (point_num + 1) * i

            self.G.add_node(new_id, pos=(lat, lon), group=group)

            clique_nodes.append(new_id)
            new_id += 1

        self.make_clique(clique_nodes, group)

    def segment_edges(self, threshold: float, seg_len: float, group: int=0):
        """
        segment the edge with length greater than threshold
        form clique for the whole edge
        """
        
        edges = list(self.G.edges(data='length'))
        for u, v, l in edges:
            if l > threshold:
                point_num = int(np.ceil(l / seg_len))
                self.segment_edge(u, v, point_num, group)

        self.update_edges()
        self.update_pairs()

    def cluster_inter_nodes(self, k: int=3, group: int=0):
        """
        merge close intermediate nodes with group == given group
        the group attribute of new nodes is also set to group
        1. cluster the intermediate nodes by k-means
        2. merge the nodes in the same cluster
        """
        
        nodes = [ node for node in self.G.nodes(data=False) 
                    if self.G.nodes[node]['group'] == group]
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
                    dist = geo.distance(center, pos).km
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
            self.G.add_node(str(new_id), pos=(lon, lat), group=group)
            new_id += 1

        self.update_edges()
        self.update_pairs()

    def update_pairs(self):
        """
        get all pairs of nodes in the network
        """
        pairs = []
        nodes = list(self.G.nodes(data=False))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pairs.append((nodes[i], nodes[j]))
                
        self.pairs = pairs
        return pairs

    def update_edges(self):
        """
        update edge attributes:
            -length
            -channel_capacity
        based on the geodesic distance between nodes
        """
        for edge in self.G.edges(data=False):
            u, v = edge
            u_pos = self.G.nodes[u]['pos']
            v_pos = self.G.nodes[v]['pos']
            length = geo.distance(u_pos, v_pos).km
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
    net = Topology(task)

    city_num = len(vset.vertices)

    net.connect_nearest_nodes(num=5, group=0)
    net.plot(None, None, './result/test/fig_cluster.png')

    net.connect_nearest_component()
    net.plot(None, None, './result/test/fig_nearest.png')
    edge_num = len(net.G.edges)
    print(edge_num)

    net.segment_edges(100, 100, 1)
    net.plot(None, None, './result/test/fig_segment.png')
    edge_num = len(net.G.edges)
    print(edge_num)

    inter_node_num = len(net.G.nodes) - city_num

    net.cluster_inter_nodes(inter_node_num // 3, 1)
    net.plot(None, None, './result/test/fig_merge.png')

    net.connect_nearest_nodes(num=5, group=2)
    net.plot(None, None, './result/test/fig_cluster2.png')

    # net.connect_nearest_component()
    # net.plot(None, None, './result/test/fig_nearest2.png')

    # net.segment_edges(100, 100)
    # net.plot(None, None, './result/test/fig_segment2.png')






    # print(net.G.nodes(data=True))
    # print(net.G.edges(data=True))
    # print(len(net.G.nodes), len(net.G.edges))


    
