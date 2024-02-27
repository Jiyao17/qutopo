
from enum import Enum
import copy
from typing import NewType

import networkx as nx
import numpy as np

from topology import Topology, ATT, IBM
from quantum import quantum_swap


# A Quantum Overlay Network (QON) is a directed multi-graph (MultiDiGraph), therefore:
# each node is uniquely defined by node_id (int)
# each edge is uniquely defined by (src, dst, key)
# each node/edge has an associated object (Node/Edge)
# that contains all the information about the node/edge
NodeID = NewType('NodeID', int)
KeyID = NewType('KeyID', int)
NodePair = NewType('NodePair', tuple[NodeID, NodeID])
EdgeTuple = NewType('EdgeTuple', tuple[NodeID, NodeID, KeyID])
Path = NewType('Path', tuple[EdgeTuple])

# Path = list[EdgeTuple]

class NodeType(Enum):
    BUFFLESS = 2
    REPEATER = 3

    BUFFERED = 101


class Node:
    def __init__(self, node_id: NodeID, node_type: NodeType,):
        self.node_id = node_id
        self.node_type = node_type

    def __str__(self):
        return str(self.node_type) + str(self.node_id)


class BufflessNode(Node):
    def __init__(self, node_id: NodeID):
        super().__init__(node_id, NodeType.BUFFLESS)


class BufferedNode(Node):
    def __init__(self, node_id: NodeID, storage: int=0):
        super().__init__(node_id, NodeType.BUFFERED)

        self.storage = storage


class Edge:
    def __init__(self, src_id: NodeID, dst_id: NodeID, key: KeyID, 
            fidelity: float=1.0, capacity: int=1400,
            ):
        # if vlink is True, then real_path is a list of EdgeTuple
        self.src_node = src_id
        self.dst_node = dst_id
        self.key = key
        self.fidelity = fidelity
        self.capacity = capacity

        self.edge_tuple = (self.src_node, self.dst_node, self.key)
        
    def __str__(self):
        desc = f'Edge {self.edge_tuple}: '
        desc += f'(fid, cap) = ({self.fidelity}, capacity={self.capacity}) '
        return desc


class VEdge(Edge):
    def __init__(self, src_id: NodeID, dst_id: NodeID, key: KeyID, 
            fidelity: float=1.0, capacity: int=1400,
            vlink: bool=False, real_path: Path=None):
        super().__init__(src_id, dst_id, key, fidelity, capacity)
        # if vlink is True, then real_path is a list of EdgeTuple
        self.vlink = vlink
        self.real_path = real_path
        
    def __str__(self):
        desc = f'Edge {self.edge_tuple}: '
        desc += f'(fid, cap, vlink) = ({self.fidelity}, capacity={self.capacity}, {self.vlink}) '
        return desc


class QN:
    """
    Basic Quantum Network (QN) class
    """

    @staticmethod
    def is_subpath(net: nx.MultiDiGraph, path0: Path, path1: Path):
        """
        Return True iff path0 is a subpath of path1
        """

        if len(path0) > len(path1):
            return False

        for i in range(len(path1)-len(path0)+1):
            if path0 == path1[i:i+len(path0)]:
                return True
        return False

    def __init__(self, topology: Topology=ATT()):
        self.topology = topology

        # real network, without virtual edges among QMs
        self.net = nx.MultiDiGraph()
        self.nodes = list(topology.nodes)
        self.adjacency = topology.adjacency

    def net_gen(self,
                storage: 'list[int]', 
                capacity: np.ndarray,
                fidelity: np.ndarray,
        ):
        """
        Generate the real network according to
        topology and given parameters
        storage: storage capacity of each nodes
        capacity: capacity of each edge, must have same shape as adjacency
        fidelity: fidelity of each edge, must have same shape as adjacency
        """

        for i, node_id in enumerate(self.nodes):
            obj=BufferedNode(node_id, storage[i])
            self.net.add_node(node_id, obj)
        
        for i, edge in enumerate(self.topology.edges):
            u, v = edge
            cap = capacity[u][v]
            fid = fidelity[u][v]
            key = 0
            edge_tuple = (edge[0], edge[1], key)
            obj = VEdge(*edge_tuple, fid, cap)
            self.net.add_edge(edge[0], edge[1], key=key, obj=obj)


class QON:
    # real quantum overlay network, without virtual edges among QMs
    class QMSelectMethod(Enum):
        RANDOM = 0
        MAX_DEGREE = 1

    @staticmethod
    def disjoint_paths(net: nx.MultiDiGraph, src: NodeID, dst: NodeID, k: int=5):
        """
        Find at most k disjoint paths from src to dst
        If no enough paths, return all paths found
        """
        paths: list[Path] = []
        # make a deep copy of the graph when processing each user pair 
        # to find disjoint paths
        tnet = copy.deepcopy(net)
        for i in range(k):
            try:
                path_nodes: list[NodeID] = nx.shortest_path(tnet, src, dst)
            except nx.NetworkXNoPath:
                break

            path: Path = []
            # remove the edges in the path
            for i in range(len(path_nodes)-1):
                u, v = path_nodes[i], path_nodes[i+1]
                # find the shortest edge
                obj_dicts: dict = tnet[u][v]
                objs: 'list[VEdge]' = [ attr_dict['obj'] for attr_dict in obj_dicts.values()]
                shortest_edge_obj = None
                shortest_edge_len = np.iinfo(np.int32).max
                for obj in objs:
                    if obj.vlink == True:
                        edge_len = len(obj.real_path)
                        if edge_len < shortest_edge_len:
                            shortest_edge_obj = obj
                            shortest_edge_len = edge_len
                    else:
                        # a real edge
                        shortest_edge_obj = obj
                        shortest_edge_len = 1
                        break
                
                edge_tuple = (u, v, shortest_edge_obj.key)
                path.append(edge_tuple)
                tnet.remove_edge(*edge_tuple)

            paths.append(tuple(path))

        return paths

    @staticmethod
    def swap_along_path(net: nx.MultiGraph, path: Path) -> tuple[float, int]:
        """
        Return (fidelity, capacity) of the given path
        """
        assert len(path) >= 1

        edge = path[0]
        edge_obj: VEdge = net.get_edge_data(*edge)['obj']
        f = edge_obj.fidelity
        min_cap = edge_obj.capacity
        for edge in path[1:]:
            edge_obj: VEdge = net.get_edge_data(*edge)['obj']
            f1 = edge_obj.fidelity
            f = quantum_swap(f, f1)

            cap = edge_obj.capacity
            if cap < min_cap:
                min_cap = cap

        return f, min_cap

    @staticmethod
    def is_vpath(net: nx.MultiDiGraph, path: Path):
        """
        Return True iff the path contains a virtual edge
        """
        for edge_tuple in path:
            obj: VEdge = net.edges[edge_tuple]['obj']
            if obj.vlink == True:
                return True
        return False

    @staticmethod
    def expand_path(net: nx.MultiDiGraph, path: Path):
        """
        Expand the path to include all the edges in the real path
        """
        expanded_path = []
        for edge_tuple in path:
            obj: VEdge = net.edges[edge_tuple]['obj']
            if obj.vlink == True:
                expanded_path.extend(obj.real_path)
            else:
                expanded_path.append(edge_tuple)
        return tuple(expanded_path)

    @staticmethod
    def is_subpath(net: nx.MultiDiGraph, path0: Path, path1: Path, expand=0):
        """
        expand: expand the path to its real edges
            0: no expansion
            1: expand path0
            2: expand path1
            3: expand both
        Return True iff path0 is a subpath of path1
        """
        expand0 = True if expand % 2 == 1 else False
        expand1 = True if expand >= 2 else False
        if expand0:
            path0 = QON.expand_path(net, copy.deepcopy(path0))
        if expand1:
            path1 = QON.expand_path(net, copy.deepcopy(path1))

        if len(path0) > len(path1):
            return False

        for i in range(len(path1)-len(path0)+1):
            if path0 == path1[i:i+len(path0)]:
                return True
        return False

    @staticmethod
    def draw(net: nx.MultiDiGraph, filename=None):
        # save the graph to file
        # or show it on screen if filename == None
        pos = nx.spring_layout(net)
        node_colors = [ 'lightcoral' if isinstance(net.nodes[node]['obj'], BufflessNode) \
                        else 'skyblue' for node in net.nodes]

        edge_style = []
        for edge in net.edges.data():
            obj: VEdge = edge[2]['obj']
            if obj.vlink == True:
                edge_style.append('dashed')
            else:
                edge_style.append('solid')

        nx.draw(net, pos, node_color=node_colors, style=edge_style, with_labels=True)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def __init__(self, topology: Topology=ATT(), QM_num: int=5):
        self.topology = topology
        self.QM_num = QM_num
        assert self.QM_num <= len(self.topology.nodes)

        self.QMs: list[NodeID] = []
        self.EUs: list[NodeID] = []
        # real network, without virtual edges among QMs
        self.rnet = nx.MultiDiGraph()
        # vnet includes virtual edges between QMs
        self.vnet = nx.MultiDiGraph()

        self.qm_pairs: 'list[NodePair]'= []
        self.qm_rpaths: 'dict[NodePair, list[Path]]' = {}
        self.qm_vpaths: 'dict[NodePair, list[Path]]' = {}

        # self._set_QMs()
        # self._graph_gen()

    def set_QMs_EUs(self, method=QMSelectMethod.RANDOM):
        """
        Set QMs, EUs, and QM pairs
        """
        if method == QON.QMSelectMethod.RANDOM:
            self.QMs = np.random.choice(self.topology.nodes, self.QM_num, replace=False)
        elif method == QON.QMSelectMethod.MAX_DEGREE:
            self.QMs = []
            for i in range(self.QM_num):
                max_degree = 0
                max_degree_id = 0
                for node_id in self.topology.nodes:
                    if node_id not in self.QMs:
                        # degree is actually degree*2
                        degree = np.sum(self.topology.adjacency[node_id]) # // 2
                        if degree > max_degree:
                            max_degree = degree
                            max_degree_id = node_id
                self.QMs.append(max_degree_id)

        self.QMs = sorted(self.QMs)
        self.EUs = [i for i in self.topology.nodes if i not in self.QMs]
        
        # find all possible QM pairs
        for qm0 in self.QMs:
            for qm1 in self.QMs:
                if qm0 != qm1:
                    self.qm_pairs.append((qm0, qm1))

    def rnet_gen(self, qm_storage=12000, capacity=(200, 1400), fidelity=(0.96, 0.99)):
        """
        Generate the real network according to
        topology and given parameters
        """

        for i in self.topology.nodes:
            if i in self.QMs:
                self.rnet.add_node(i, obj=BufferedNode(i, qm_storage))
            else:
                self.rnet.add_node(i, obj=BufflessNode(i))
        
        for i, edge in enumerate(self.topology.edges):
            cap = np.random.randint(capacity[0], capacity[1])
            fid = np.random.uniform(fidelity[0], fidelity[1])
            key = self.rnet.number_of_edges(edge[0], edge[1])
            edge_tuple = (edge[0], edge[1], key)
            obj = VEdge(*edge_tuple, fid, cap)
            self.rnet.add_edge(edge[0], edge[1], key=key, obj=obj)

    def vnet_gen(self, qm_rpath_num=5):
        """
        Generate virtual network
        1. find disjoint (by edge) real paths for each QM pair
        2. generate vnet by adding virtual edges for each real qm path
        3. set QM vpaths (each vpath only contains the virtual edge between the QM pair)
        """

        for qm_pair in self.qm_pairs:
            paths = QON.disjoint_paths(self.rnet, qm_pair[0], qm_pair[1], qm_rpath_num)
            self.qm_rpaths[qm_pair] = paths

        self.vnet = copy.deepcopy(self.rnet)

        # add virtual edges for each path between a QM pair
        for qm_pair in self.qm_pairs:
            self.qm_vpaths[qm_pair] = []
            for path in self.qm_rpaths[qm_pair]:
                fid, cap = QON.swap_along_path(self.rnet, path)
                key = self.vnet.number_of_edges(*qm_pair)
                edge_tuple = (qm_pair[0], qm_pair[1], key)
                obj = VEdge(*edge_tuple, fid, cap, True, path)
                self.vnet.add_edge(*edge_tuple, obj=obj,)

                vpath: Path = (edge_tuple, )
                self.qm_vpaths[qm_pair].append(vpath)


class Task:
    def __init__(self) -> None:
        pass


class QONTask:

    def __init__(self, qon: QON,  time_slots: int=10, delta: float=20.0):
        self.qon = qon
        assert time_slots > 1
        self.time_slots = time_slots
        self.delta = delta
        self.user_pairs: 'list[NodePair]' = []
        # all real paths between user pairs
        self.up_rpaths: 'dict[NodePair, list[Path]]' = {}
        # paths that contain at least one virtual link
        self.up_vpaths: 'dict[NodePair, list[Path]]' = {}
        # D^k_t in the paper, set in workload_gen()
        self.workload: 'dict[tuple[NodePair, int], int]' = {}
        # F^k_t in the paper, set in workload_gen()
        self.fid_req: 'dict[tuple[NodePair, int], float]' = {}

    def set_user_pairs(self, pair_num=6, method='random'):
        # all possible edge user pairs
        user_pairs: list[NodePair] = []
        EUs = self.qon.EUs
        for i in range(len(EUs)):
            for j in range(i+1, len(EUs)):
                user_pairs.append((EUs[i], EUs[j]))
            
        if pair_num > len(user_pairs):
            raise ValueError('pair_num must be <= the number of all possible user pairs')

        # selected user pairs
        if method == 'random':
            up_indices = np.random.choice(len(user_pairs), pair_num, replace=False)
            self.user_pairs = [user_pairs[idx] for idx in up_indices]

    def set_up_paths(self, path_num=3):
        """
        find disjoint (by edge) real & virtual paths for each user pair
        """

        for user_pair in self.user_pairs:
            paths = QON.disjoint_paths(self.qon.vnet, *user_pair, path_num)
            self.up_rpaths[user_pair] = []
            self.up_vpaths[user_pair] = []
            for path in paths:
                if QON.is_vpath(self.qon.vnet, path):
                    self.up_vpaths[user_pair].append(path)
                else:
                    self.up_rpaths[user_pair].append(path)
        
    def workload_gen(self, request_range=(1, 100), fid_range=(0.8, 0.8001)):
        
        for i, user_pair in enumerate(self.user_pairs):
            # generate a random load for each user pair
            load = np.random.randint(*request_range)
            self.workload[(user_pair, 0)] = 0
            self.fid_req[(user_pair, 0)] = 0

        # generate D^k_t for each user pair, t >= 1
        for t in range(1, self.time_slots):
            for i, user_pair in enumerate(self.user_pairs):
                # generate a random load for each user pair
                load = np.random.randint(*request_range)
                self.workload[(user_pair, t)] = load
                # generate a random fid for each user pair
                fid = np.random.uniform(*fid_range)
                self.fid_req[(user_pair, t)] = fid
        
        # generate fid_req for each storage pair, t >= 1
        # this is undefined in the paper, but we need for optimization
        # due to constraint 5 (left side)
        # if not set, always get g = 0, then w can be infinite
        for qm_pair in self.qon.qm_pairs:
            self.fid_req[(qm_pair, 0)] = 0
        for t in range(1, self.time_slots):
            for qm_pair in self.qon.qm_pairs:
                fid = np.random.uniform(*fid_range)
                self.fid_req[(qm_pair, t)] = fid


def test():
    import matplotlib.pyplot as plt
    net = nx.Graph()

    QM_num = 0
    EU_num = 7
    QMs = []
    EUs = []

    # QM_id_list = np.random.choice(range(QM_num + EU_num), QM_num, replace=False)
    # QM_id_list = [2, 3, 9]
    QM_id_list = []
    for i in range(QM_num + EU_num):
        if i in QM_id_list:
            net.add_node(i, obj=Node(i, NodeType.BUFFERED))
        else:
            net.add_node(i, obj=Node(i, NodeType.BUFFLESS))
    
    edges = [
        (0, 1), (1, 2), (1,3), (2, 4), (2, 5),
        (3, 5), (4, 5), (4, 6), 
        
    ]

    edge_colors = []
    red_edges = []
    blue_edges = []
    green_edges = []
    for edge in edges:
        obj = VEdge(edge[0], net.nodes[edge[0]]['obj'], net.nodes[edge[1]]['obj'])
        if edge in red_edges:
            edge_colors.append('red')
        elif edge in blue_edges:
            edge_colors.append('blue')
        elif edge in green_edges:
            edge_colors.append('green')
        else:
            edge_colors.append('black')
        net.add_edge(edge[0], edge[1], obj=obj)

    for edge in edges:
        net.add_edge(edge[0], edge[1])

    subax1 = plt.subplot(121)
    colors = ['orange' if i in QM_id_list else 'green' for i in range(QM_num + EU_num)]
    nx.draw(net, ax=subax1, with_labels=True, node_color=colors, edge_color=edge_colors, )
    plt.savefig('test.png')


def test_graph():
    # draw an example graph
    # where QM nodes are orange, EU nodes are green


    import matplotlib.pyplot as plt
    np.random.seed(0)

    qon = QON(topology=ATT(), QM_num=5)
    qon.set_QMs_EUs(QON.QMSelectMethod.MAX_DEGREE)
    qon.rnet_gen()
    qon.vnet_gen()
    print("All QM paths: \n", qon.qm_rpaths)
    # check the fidelity of the virtual links
    for edge in qon.vnet.edges:
        edge_obj: VEdge = qon.vnet.edges[edge]['obj']
        if edge_obj.vlink == True:
            fid = edge_obj.fidelity
            fid_calc = QON.swap_along_path(qon.vnet, edge_obj.real_path)[0]
            assert fid == fid_calc, "Fidelity calculation error!"

    task = QONTask(qon)
    task.set_user_pairs(5)
    # task.user_pairs.append((7, 21))
    task.user_pairs.append((10, 14))
    print("All user pairs: \n", task.user_pairs)
    task.set_up_paths()
    print("All rpaths between selected user pairs: \n", task.up_rpaths)

    # check fidelity of the paths [12, 15, 1, 16]
    # edge12_15: Edge = qon.net.edges[12, 15]['obj']
    # edge15_1: Edge = qon.net.edges[15, 1]['obj']
    # edge1_16: Edge = qon.net.edges[1, 16]['obj']
    # print(edge12_15.fidelity, edge15_1.fidelity, edge1_16.fidelity)
    # f = quantum_swap(edge12_15.fidelity, edge15_1.fidelity)
    # f = quantum_swap(f, edge1_16.fidelity)
    # print(f)


    # print all edges to check the paths
    # remember that sometimes the edges overlap, or no edges at all
    # so pls check here to make sure the paths are correct
    # instead of the only verify on the shown graph
    # print("All edges: \n", task.qon.vnet.edges)

    QON.draw(qon.rnet)




if __name__ == "__main__":
    test()