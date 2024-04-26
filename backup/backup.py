
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

