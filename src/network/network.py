
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
    'pm_install': 1e2, # memory installation price
    'pc': 1, # channel price per km
    # 'pc_install': 1e4, # channel installation price
    'pc_install': 1e2, # channel installation price
}


class IDGenerator:
    
        def __init__(self, start: int=0) -> None:
            self.start = start - 1
    
        def __iter__(self):
            return self
    
        def __next__(self):
            self.start += 1
            return self.start


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
            'lat_min': min([v[0] for v in self.U.values()]),
            'lat_max': max([v[0] for v in self.U.values()]),
            'lon_min': min([v[1] for v in self.U.values()]),
            'lon_max': max([v[1] for v in self.U.values()]),
            }
        
        self.graph: 'nx.Graph' = nx.Graph()
        self.nid = IDGenerator()
        for node in self.U.keys():
            pos = self.U[node]
            self.graph.add_node(next(self.nid), pos=pos, group=0)
        # for edge, length in self.task.vset.edges.items():
        #     src, dst = edge
        #     length = geo.geodesic(self.U[src], self.U[dst]).km
        #     self.graph.add_edge(src, dst, length=length, group=0)

        self.pairs = self.update_pairs()

    def connect_nodes_nearest(self, num: int=5, group: int=0):
        """
        cluster nearby nodes
        """
        # connect nearest num nodes
        
        for u in self.graph.nodes(data=False):
            nearest_found = []
            for v in self.graph.nodes(data=False):
                if u != v :
                    u_pos = self.graph.nodes[u]['pos']
                    v_pos = self.graph.nodes[v]['pos']
                    distance = geo.distance(u_pos, v_pos).km
                    if len(nearest_found) < num:
                        nearest_found.append((v, distance))
                    else:
                        nearest_found.sort(key=lambda x: x[1])
                        if distance < nearest_found[-1][1]:
                            nearest_found[-1] = (v, distance)
            for v, _ in nearest_found:
                self.graph.add_edge(u, v, length=distance, group=group)

        self.update_edges()

    def connect_nodes_radius(self, radius: float=200, group: int=0):
        """
        cluster nearby nodes within radius
        """
        # connect any two nodes within radius
        for u in self.graph.nodes(data=False):
            for v in self.graph.nodes(data=False):
                if u != v :
                    u_pos = self.graph.nodes[u]['pos']
                    v_pos = self.graph.nodes[v]['pos']
                    distance = geo.distance(u_pos, v_pos).km
                    if distance <= radius:
                        self.graph.add_edge(u, v, length=distance, group=group)

        self.update_edges()

    def connect_nearest_component(self, group: int=0):
        """
        connect the graph by repeatedly connecting the nearest components
        """
        components = list(nx.connected_components(self.graph))
        while len(components) > 1:
            min_dist = np.inf
            min_edge = None
            for i in range(len(components) - 1):
                for j in range(i + 1, len(components)):
                    # find the nearest pair of nodes
                    for u in components[i]:
                        for v in components[j]:
                            u_pos = self.graph.nodes[u]['pos']
                            v_pos = self.graph.nodes[v]['pos']
                            distance = geo.distance(u_pos, v_pos).km
                            if distance < min_dist:
                                min_dist = distance
                                min_edge = (u, v)
            self.graph.add_edge(*min_edge, length=min_dist, group=group)
            components = list(nx.connected_components(self.graph))

        self.update_edges()

    def make_clique(self, nodes: list, group: int=0):
        """
        form a clique for the nodes
        """
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u = nodes[i]
                v = nodes[j]
                u_pos = self.graph.nodes[u]['pos']
                v_pos = self.graph.nodes[v]['pos']
                distance = geo.distance(u_pos, v_pos).km
                self.graph.add_edge(u, v, length=distance, group=group)

        self.update_edges()

    def segment_edge_line(self, u, v, point_num, group=0):
        """
        segment the edge with length greater than threshold
        form clique for the whole edge
        """
        u_lat, u_lon = self.graph.nodes[u]['pos']
        v_lat, v_lon = self.graph.nodes[v]['pos']
        nodes = []
        # add seg_num nodes between u and v
        for i in range(1, point_num + 1):
            lat = u_lat + (v_lat - u_lat) / (point_num + 1) * i
            lon = u_lon + (v_lon - u_lon) / (point_num + 1) * i

            new_id = next(self.nid)
            self.graph.add_node(new_id, pos=(lat, lon), group=group)
            nodes.append(new_id)

        nodes = [u, ] + nodes + [v, ]
        for i in range(len(nodes) - 1):
            p, q = nodes[i], nodes[i + 1]
            p_pos = self.graph.nodes[p]['pos']
            q_pos = self.graph.nodes[q]['pos']
            distance = geo.distance(p_pos, q_pos).km
            self.graph.add_edge(p, q, length=distance, group=group)    

    def segment_edge_clique(self, u, v, point_num, group=0):
        """
        segment the edge with length greater than threshold
        form clique for the whole edge
        """
        u_lat, u_lon = self.graph.nodes[u]['pos']
        v_lat, v_lon = self.graph.nodes[v]['pos']
        clique_nodes = [u, v]
        # add seg_num nodes between u and v
        for i in range(1, point_num + 1):
            lat = u_lat + (v_lat - u_lat) / (point_num + 1) * i
            lon = u_lon + (v_lon - u_lon) / (point_num + 1) * i

            new_id = next(self.nid)
            self.graph.add_node(new_id, pos=(lat, lon), group=group)

            clique_nodes.append(new_id)

        self.make_clique(clique_nodes, group)

    def segment_edges(self, threshold: float, seg_len: float, group: int=0):
        """
        segment the edge with length greater than threshold
        form clique for the whole edge
        """
        
        edges = list(self.graph.edges(data=True))
        for u, v, d in edges:
            if d['length'] > threshold and d['group'] == group:
                point_num = int(np.ceil(d['length'] / seg_len))
                self.segment_edge_line(u, v, point_num, group)

        self.update_edges()
        self.update_pairs()

    def cluster_inter_nodes(self, k: int=3, groups: set={1}):
        """
        merge close intermediate nodes with group in groups
        the group attribute of new nodes is also set to group
        1. cluster the intermediate nodes by k-means
        2. merge the nodes in the same cluster
        """
        
        nodes = [ node for node in self.graph.nodes(data=False) 
                    if self.graph.nodes[node]['group'] in groups]
        if k > len(nodes):
            k = len(nodes)
        if k == 0:
            return
        # pick k random nodes as initial cluster centers
        initial_center_nodes = np.random.choice(nodes, k, replace=False)
        centers = np.array([self.graph.nodes[node]['pos'] for node in initial_center_nodes])
        clusters: 'list[list]'= [ [ ] for _ in range(k)]
        new_centers = np.zeros((k, 2))
        # k-means
        while True:
            for node in nodes:
                pos = self.graph.nodes[node]['pos']
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
                    new_centers[i] = np.mean([self.graph.nodes[node]['pos'] for node in cluster], axis=0)
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
            clusters = [ [ ] for _ in range(k)]
            
        # remove all intermediate nodes
        for node in nodes:
            self.graph.remove_node(node)
        # add new nodes
        for i, cluster in enumerate(clusters):
            self.graph.add_node(next(self.nid), pos=centers[i], group=max(groups) + 1)


        self.update_edges()
        self.update_pairs()

    def add_nodes_random(self, num: int, group: int=0):
        """
        add random points to the network
        """
        for i in range(num):
            lat = np.random.uniform(self.area['lat_min'], self.area['lat_max'])
            lon = np.random.uniform(self.area['lon_min'], self.area['lon_max'])
            self.graph.add_node(next(self.nid), pos=(lat, lon), group=group)

        self.update_pairs()

    def get_grid_points(self, area: dict, width: float):
        """
        generate grids of size*size in the area
        """
        nodes = {}
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
        
        center = ((area['lat_max'] + area['lat_min']) / 2, (area['lon_max'] + area['lon_min']) / 2)
        hstep = int(np.ceil(hlen_km / width))
        vstep = int(np.ceil(vlen_km / width))
        
        # vertical center line
        vnodes = [center, ]
        nodes[next(self.nid)] = center
        for i in range(1, int(np.ceil(vstep / 2)) + 1):
            vnode = (center[1] + i * vlen / vstep, center[0])
            vnodes.append(vnode)
            nodes[next(self.nid)] = vnode
            vnode = (center[1] - i * vlen / vstep, center[0])
            vnodes.append(vnode)
            nodes[next(self.nid)] = vnode
        # horizontal expansion
        for vnode in vnodes:
            for i in range(1, int(np.ceil(hstep / 2)) + 1):
                lat = vnode[1]
                lon = vnode[0] + i * hlen / hstep
                nodes[next(self.nid)] = (lat, lon)
                lon = vnode[0] - i * hlen / hstep
                nodes[next(self.nid)] = (lat, lon)

        return nodes

    def add_grid_points(self, area: dict=None, width: float=200, group: int=0):
        """
        add random points to the network
        """
        if area is None:
            area = self.area

        nodes = self.get_grid_points(area, width)
        for node_id in nodes.keys():
            self.graph.add_node(node_id, pos=nodes[node_id], group=group)

        # self.nodes = self.G.nodes(data=False)
        self.update_pairs()

    def update_pairs(self):
        """
        get all pairs of nodes in the network
        """
        pairs = []
        nodes = list(self.graph.nodes(data=False))
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
        for edge in self.graph.edges(data=False):
            u, v = edge
            u_pos = self.graph.nodes[u]['pos']
            v_pos = self.graph.nodes[v]['pos']
            length = geo.distance(u_pos, v_pos).km
            fiber_loss = self.hw_params['fiber_loss']
            photon_rate = self.hw_params['photon_rate']
            # prob for half fiber (detectors are in the middle of edges)
            prob = 10 ** (-0.1 * fiber_loss * (length/2)) 
            channel_capacity = photon_rate * prob**2
            # round to 2 decimal places
            self.graph[u][v]['length'] = np.round(length, 2)
            self.graph[u][v]['channel_capacity'] = channel_capacity

    def plot(self, 
            node_label: str='id', 
            edge_label: str='length', 
            filename: str='./result/fig.png'
            ):
        plot_nx_graph(self.graph, node_label, edge_label, filename)




if __name__ == '__main__':
    topo = VertexSource.GETNET
    vset = VertexSet(topo)
    task = Task(vset)
    net = Topology(task)

    city_num = len(vset.vertices)

    net.connect_nodes_nearest(num=10, group=1)
    net.plot(None, None, './result/test/fig_cluster.png')

    net.connect_nearest_component(1)
    net.plot(None, None, './result/test/fig_nearest.png')
    edge_num = len(net.graph.edges)
    print(edge_num)

    net.segment_edges(200, 200, 1)
    net.plot(None, None, './result/test/fig_segment.png')
    edge_num = len(net.graph.edges)
    print(edge_num)

    inter_node_num = len(net.graph.nodes) - city_num
    print(inter_node_num)


    round_num = 10
    for i in range(1, round_num + 1):

        net.add_nodes_random(inter_node_num // 3, i)
        net.plot(None, None, f'./result/test/fig_random_{i}.png')

        net.cluster_inter_nodes(inter_node_num // 3, set(range(1, i + 1)))
        net.plot(None, None, f'./result/test/fig_merge_{i}.png')

    # net.connect_nodes_nearest(num=5, group=2)
    # net.plot(None, None, './result/test/fig_cluster2.png')

    # net.connect_nearest_component()
    # net.plot(None, None, './result/test/fig_nearest2.png')

    # net.segment_edges(100, 100)
    # net.plot(None, None, './result/test/fig_segment2.png')






    # print(net.G.nodes(data=True))
    # print(net.G.edges(data=True))
    # print(len(net.G.nodes), len(net.G.edges))


    
