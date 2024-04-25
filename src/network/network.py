
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

    def add_grid_points(self, area: dict=None, width: float=200):
        """
        add random points to the network
        """
        if area is None:
            area = self.area

        nodes = self.get_grid_points(area, width)
        for node_id in nodes.keys():
            lon, lat = nodes[node_id]
            self.G.add_node(node_id, pos=(lon, lat), original=False)

        # self.nodes = self.G.nodes(data=False)
        # self.update_pairs()
    
    def get_grid_points(self, area: dict, width: float):
            """
            generate grids of size*size in the area
            """
            nodes = {}
            new_id = len(self.G.nodes)
            hlen_km = geo.distance(
                (area['lat_min'], area['lon_min']),
                (area['lat_min'], area['lon_max'])
            ).km
            vlen_km = geo.distance(
                (area['lat_min'], area['lon_min']),
                (area['lat_max'], area['lon_min'])
            ).km
            hlen = area['lon_max'] - area['lon_min']
            vlen = area['lat_max'] - area['lat_min']
            center = ((area['lon_max'] + area['lon_min']) / 2, (area['lat_max'] + area['lat_min']) / 2)
            hstep = int(np.ceil(hlen_km / width))
            vstep = int(np.ceil(vlen_km / width))
            
            # vertical center line
            vnodes = [center, ]
            nodes[str(new_id)] = center
            new_id += 1
            for i in range(1, int(np.ceil(vstep / 2)) + 1):
                vnode = (center[0], center[1] + i * vlen / vstep)
                vnodes.append(vnode)
                nodes[str(new_id)] = vnode
                new_id += 1
                vnode = (center[0], center[1] - i * vlen / vstep)
                vnodes.append(vnode)
                nodes[str(new_id)] = vnode
                new_id += 1
            # horizontal expansion
            for vnode in vnodes:
                for i in range(1, int(np.ceil(hstep / 2)) + 1):
                    lon = vnode[0] + i * hlen / hstep
                    lat = vnode[1]
                    nodes[str(new_id)] = (lon, lat)
                    new_id += 1
                    lon = vnode[0] - i * hlen / hstep
                    nodes[str(new_id)] = (lon, lat)
                    new_id += 1

            return nodes

    def mst(self, prune: bool=False):
        """
        form the minimum spanning tree
        """
        # find the minimum spanning tree over original nodes

        mst = nx.minimum_spanning_tree(self.G, weight='length')
        if prune:
            self.G = mst

        return mst

    def cluster_by_distance(self, threshold: float=100):
        """
        cluster nearby nodes
        """
        # connect any two nodes within threshold
        for u in self.G.nodes(data=False):
            for v in self.G.nodes(data=False):
                if u != v :
                    pos1 = self.G.nodes[u]['pos']
                    pos2 = self.G.nodes[v]['pos']
                    distance = geo.distance((pos1[1], pos1[0]), (pos2[1], pos2[0])).km
                    if distance < threshold:
                        self.G.add_edge(u, v, length=distance)

        self.update_edges()

    def cluster_by_nearest(self, num: int=3):
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

    def connect_by_mst(self):
        mst: nx.Graph = nx.minimum_spanning_tree(self.G)
        components = list(nx.connected_components(self.G))
        while len(components) > 1:
            for edge in mst.edges:
                c0, c1 = None, None
                for i in range(len(components)):
                    for node in components[i]:
                        if node == edge[0]:
                            c0 = i
                        if node == edge[1]:
                            c1 = i
                if c0 != c1:
                    self.G.add_edge(*edge)
                    components = list(nx.connected_components(self.G))
                    break
        
        self.update_edges()

    def make_clique_among_components(self):
        """
        add edges between every pair of components
        """
        components = list(nx.connected_components(self.G))
        # connect the components by adding an shortest edge
        for i in range(len(components) - 1):
            for j in range(i + 1, len(components)):
                min_dist = np.inf
                min_edge = None
                for u in components[i]:
                    for v in components[j]:
                        pos1 = self.G.nodes[u]['pos']
                        pos2 = self.G.nodes[v]['pos']
                        distance = geo.distance((pos1[1], pos1[0]), (pos2[1], pos2[0])).km
                        if distance < min_dist:
                            min_dist = distance
                            min_edge = (u, v)
                self.G.add_edge(*min_edge, length=min_dist)

    def nearest_components(self):
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

    def make_clique(self, nodes=None):
        """
        make a clique network for given nodes
        """
        if nodes is None:
            nodes = self.G.nodes(data=False)
        # add edges if not exist
        for node in nodes:
            for neighbor in nodes:
                if node != neighbor and not self.G.has_edge(node, neighbor):
                    self.G.add_edge(node, neighbor)

        self.update_edges()

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
    
    def plot(self, 
            node_label: str='id', 
            edge_label: str='length', 
            filename: str='./result/fig.png'
            ):
        plot_nx_graph(self.G, node_label, edge_label, filename)


if __name__ == '__main__':
    topo = VertexSource.ATT
    vset = VertexSet(topo)
    task = Task(vset)
    net = Network(task)

    # net.cluster(300)
    # net.plot(None, None, './result/fig_cluster.png')

    # net.add_grid_points(width=500)
    # net.plot(None, None, './result/fig_grid.png')

    net.cluster_by_nearest(num=1)
    net.plot(None, None, './result/fig_cluster.png')

    net.nearest_components()
    net.plot(None, None, './result/fig_nearest.png')

    # net.connect_by_mst()
    # net.plot(None, None, './result/fig_connect.png')

    # net.make_clique_among_components()
    # net.plot(None, None, './result/fig_clique.png')
    
    # net.segment_edge(200, 200)
    # net.plot(None, None, './result/fig_seg.png')

    # net.make_clique()
    # net.plot(None, None, './result/fig_clique.png')

    # net.mst(prune=True)
    # net.plot(None, None, './result/fig_mst.png')
    

    # net.add_points_random(num=2)
    # net.make_clique()

    # print(net.G.nodes(data=True))
    # print(net.G.edges(data=True))
    # print(len(net.G.nodes), len(net.G.edges))


    
