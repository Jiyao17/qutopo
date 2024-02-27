
import copy
import numpy as np
import enum
import time

import numpy as np
import gurobipy as gp

from net import EdgeTuple
from quantum import swap_dephased, swap_depolarized, purify_dephased, purify_depolarized
from quantum import swap_dephased_grad, swap_depolarized_grad, purify_dephased_grad, purify_depolarized_grad



class SSSolver:
    """
    Swap Scheme Solver
    """

    def __init__(self, edges: dict[EdgeTuple, float]) -> None:
        # each node in the SS Tree is an edge in the graph
        # dict[edge_tuple, fidelity]
        self.edges = copy.deepcopy(edges)


class IPSolver(SSSolver):
    """
    Integer Programming Solver for Swap Scheme
    """

    def __init__(self, edges: dict[EdgeTuple, float]) -> None:
        super().__init__(edges)

        self.model = gp.Model("IP")
        self.N = len(edges) + 1 # number of nodes
        self.T = self.N - 2 # N-2 swaps
        self.M: 'dict[tuple[int], gp.Var]' = {}
        self.A: 'dict[tuple[int], gp.LinExpr]' = {}

        self.NodeRange = range(self.N)
        self.SwapTimeRange = range(self.T)
        self.FidelityTimeRange = range(self.T+1)

    def build(self) -> None:
        def add_vars():
            """
            add variables
            """
            edges = [edge for edge in self.edges.keys()]
            nodes = [edge[0] for edge in edges]
            nodes.append(edges[-1][1])
            M: list[tuple] = []
            for t in self.SwapTimeRange:
                for k in self.NodeRange:
                    M.append((t, k))

            for t in self.FidelityTimeRange:
                for i in self.NodeRange:
                    for j in self.NodeRange:
                        self.A[(t, i, j)] = 0
            for edge in self.edges.keys():
                self.A[(0, edge[0], edge[1])] = self.edges[edge]
                self.A[(0, edge[1], edge[0])] = self.edges[edge]

            self.M = self.model.addVars(M, vtype=gp.GRB.BINARY, name='M')


        def add_constr():
            # each intermediate node should be swapped exactly once
            for k in self.NodeRange[1:-1]:
                self.model.addConstr(gp.quicksum(self.M[t, k] for t in self.SwapTimeRange) == 1,
                                        name=f'M_sum_{k}')

            # should not swap at node 0 and N
            for t in self.SwapTimeRange:
                self.model.addConstr(self.M[t, 0] == 0, name=f'M_{t}_0')
                self.model.addConstr(self.M[t, self.NodeRange[-1]] == 0,
                                     name=f'M_{t}_{self.NodeRange[-1]}')
                
            # at each time step, exactly one swap should be performed
            for t in self.SwapTimeRange:
                self.model.addConstr(gp.quicksum(self.M[t, k] for k in self.NodeRange[1:-1]) == 1,
                                        name=f'M_{t}_sum')
    
            I = {}
            for t in self.FidelityTimeRange[1:]:
                for i in self.NodeRange:
                    for j in self.NodeRange[i+1:]:
                        print(f'add fidelity constr: {t}, {i}, {j}')
                        F = 0
                        for k in self.NodeRange:
                            if i == k or j == k:
                                continue
                            deno = 0
                            for p in self.NodeRange:
                                for q in self.NodeRange[p+1:]:
                                    deno += self.A[t-1, p, k] * self.A[t-1, k, q]
                            S = quantum_swap(self.A[t-1, i, k], self.A[t-1, j, k])
                            F += self.M[t-1, k] * self.A[t-1, p, k] * self.A[t-1, k, q] * S / (deno+0.00000001)
                            # F += self.M[t-1, k] * S
                        
                        # copy or update
                        selected = (self.M[t-1, i] or self.M[t-1, j]) and (i, j) != (0, self.NodeRange[-1])
                        not_selected = not selected
                        self.A[t, i, j] =  self.A[t-1, i, j] * not_selected \
                                            + selected * F
                        self.A[t, j, i] = self.A[t, i, j]

                        # I[(t, i, j)] = F
                        # self.A[(t, i, j)] = I[(t, i, j)]
                        # self.A[(t, j, i)] = I[(t, i, j)]
                        # self.model.addConstr(self.A[(t, i, j)] >= 0, name=f'A_{t}_{i}_{j}')
                        # self.model.addConstr(self.A[(t, p, q)] >= 0, name=f'A_{t}_{q}_{p}')

                            # self.model.addConstr(self.A[(t, p, q)] == self.M[t-1, k] * self.A[t-1, p, k] * self.A[t-1, q, k],
                                                #  name=f'A_{t}_{p}_{q}')

                        # self.model.addConstr(self.A[(t, q, p)] == self.A[(t, p, q)], name=f'A_{t}_{q}_{p}')
                       
        add_vars()
        add_constr()
        self.model.update()
                
    def solve(self) -> None:
        """
        solve the problem using integer programming
        """
        s, t = self.NodeRange[0], self.NodeRange[-1]
        last_time = self.FidelityTimeRange[-1]
        self.model.setObjective(self.A[last_time, s, t], gp.GRB.MAXIMIZE)
        self.model.update()
        # model size
        print(f"model size: {self.model.NumVars}, {self.model.NumConstrs}")
        self.model.optimize()
        return self.model.ObjVal


class SPSSolver:
    """
    Swap Purification Scheme Solver
    """

    def __init__(self, edges: dict[EdgeTuple, float]) -> None:
        # each node in the SPS Tree is an edge in the graph
        # dict[edge_tuple, fidelity]
        self.edges = copy.deepcopy(edges)


class Node:
    node_id = -1
    @staticmethod
    def get_new_id():
        Node.node_id += 1
        return Node.node_id

    def __init__(self, node_id: int, edge_tuple: EdgeTuple,
                parent, left, right, fidelity: float) -> None:
        self.node_id: int = node_id
        self.edge_tuple: EdgeTuple = edge_tuple

        self.parent: Node = parent
        self.left: Node = left
        self.right: Node = right

        self.fidelity: float = fidelity
        self.grad: float = 1
        self.efficiency: float = 1
        self.adjust: float = 1
        self.adjust_eff: float = 1
        self.test_rank_attr: float = 1

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def is_root(self) -> bool:
        return self.parent is None

    def __str__(self) -> str:
        s = f'{self.node_id} {self.edge_tuple}: '
        if self.parent is None:
            s += f'f={self.fidelity}, '
        # keep 2 decimal places
        else:
            s += f'f={self.fidelity:.2f}, '

        s += f'g={self.grad:.5f}, e={self.efficiency:.5f}, a={self.adjust:.5f}, ae={self.adjust_eff:.5f}'
        return s


class Leaf(Node):
    """
    Leaf Node in the SPS Tree
    """

    def __init__(self, node_id: int, edge_tuple: EdgeTuple,
                 parent: Node, fidelity: float) -> None:
        super().__init__(node_id, edge_tuple, parent, None, None, fidelity)

    def __str__(self) -> str:
        # keep 2 decimal places
        return f'Leaf ' + super().__str__()


class BranchOp(enum.Enum):
    SWAP = 0
    PURIFY = 1


class Branch(Node):
    """
    Branch Node in the SPS Tree
    """

    def __init__(self, node_id: int, edge_tuple: EdgeTuple,
                  parent: Node, left: Node, right: Node,
                    fidelity: float, op: BranchOp.SWAP) -> None:
        super().__init__(node_id, edge_tuple, parent, left, right, fidelity)
        # -1 indicates that there is not a real edge in the graph
        self.edge_tuple = edge_tuple[0], edge_tuple[1], -1

        self.op: BranchOp = op
    
    def __str__(self) -> str:
        # keep 2 decimal places
        return f'Branch ' + super().__str__()


class SystemType(enum.Enum):
    DEPHASED = 0
    DEPOLARIZED = 1


class MetaTree:
    
    def __init__(self, leaves: 'list', system_type: SystemType.DEPHASED) -> None:
        assert len(leaves) > 0, 'edges must be more than 0'
        
        self.leaves: list = leaves
        self.system_type = system_type
        self.root: Branch = None

        if self.system_type == SystemType.DEPHASED:
            self.swap_func = swap_dephased
            self.swap_grad = swap_dephased_grad
            self.purify_func = purify_dephased
            self.purify_grad = purify_dephased_grad
        else:
            self.swap_func = swap_depolarized
            self.swap_grad = swap_depolarized_grad
            self.purify_func = purify_depolarized
            self.purify_grad = purify_depolarized_grad


class SSTree(MetaTree):
    def __init__(self, leaves: list, system_type: SystemType.DEPHASED) -> None:
        super().__init__(leaves, system_type)


class PSTree(MetaTree):
    def __init__(self, leaves: list, system_type: SystemType.DEPHASED) -> None:
        super().__init__(leaves, system_type)


class SPSTR(MetaTree):
    def __init__(self, leaves: list, system_type: SystemType.DEPHASED) -> None:
        super().__init__(leaves, system_type)


class SPSTree:
    """
    Swap Purification Scheme Tree
    SPSTree is a binary tree becasue:
    both swap and purification are binary operators
    """

    @staticmethod
    def print_tree(root=None, indent=0):
        root: Node = root # type hinting here
        if root is None:
            return
        print('  ' * indent + str(root))
        SPSTree.print_tree(root.left, indent + 1)
        SPSTree.print_tree(root.right, indent + 1)

    @staticmethod
    def traverse(root: Node=None):
        """
        traverse the tree in mid-order
        print all leaf nodes

        Use this function to check if SST (not SPST) is legal
        """
        if root is None:
            return
        SPSTree.traverse(root.left)
        if root.is_leaf():
            s = str(root.node_id)
            print(s, end=' ')
        SPSTree.traverse(root.right)

        if root.parent is None:
            print()

    @staticmethod
    def copy_subtree(node: Node) -> Node:
        if node is None:
            return None
        # new_node = Node(node.node_id, node.parent, None, None, node.fidelity)
        # new_node.left = copy_subtree(node.left)
        # new_node.right = copy_subtree(node.right)
        new_node = copy.deepcopy(node)
        return new_node

    @staticmethod
    def get_height(node: Node) -> int:
        if node is None:
            return 0
        return max(SPSTree.get_height(node.left), SPSTree.get_height(node.right)) + 1

    @staticmethod
    def count_leaves(node: Node) -> int:
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return SPSTree.count_leaves(node.left) + SPSTree.count_leaves(node.right)

    @staticmethod
    def find_max(node: Node, attr: str="adjust_eff", search_range=[Node]) -> Node:
        """
        find the node with max attr in the subtree
        attr: must be a member of Node
        """
        def in_range(node: Node, range: 'list') -> bool:
            for type in range:
                if isinstance(node, type):
                    return True
            return False

        if node is None:
            return None
        if node.is_leaf():
            if in_range(node, search_range):
                return node
            else:
                return None
        
        candidates = []
        if in_range(node, search_range):
            candidates.append(node)
        left = SPSTree.find_max(node.left, attr, search_range)
        if in_range(left, search_range):
            candidates.append(left)
        right = SPSTree.find_max(node.right, attr, search_range)
        if in_range(right, search_range):
            candidates.append(right)

        if len(candidates) == 0:
            return None
        
        max_node = candidates[0]
        for node in candidates:
            if getattr(node, attr) > getattr(max_node, attr):
                max_node = node        
        return max_node

        # if getattr(left, attr) > getattr(right, attr):
        #     return left
        # else:
        #     return right

    def __init__(self, edges: dict[EdgeTuple, float], system_type: SystemType.DEPHASED) -> None:
        assert len(edges) > 0, 'edges must be more than 0'
        # id -> edge_tuple
        # used to interpret SPS results
        self.leaves: 'dict[int, EdgeTuple]' = {}
        # id -> node object
        # used to point a node in the tree quickly
        self.nodes: 'dict[int, Node]' = {}

        for edge, fid in edges.items():
            new_id = Node.get_new_id()
            self.leaves[new_id] = edge
            self.nodes[new_id] = Leaf(new_id, edge, None, fid)
        
        self.system_type = system_type
        self.root: Branch = None

        if self.system_type == SystemType.DEPHASED:
            self.swap_func = swap_dephased
            self.swap_grad = swap_dephased_grad
            self.purify_func = purify_dephased
            self.purify_grad = purify_dephased_grad
        else:
            self.swap_func = swap_depolarized
            self.swap_grad = swap_depolarized_grad
            self.purify_func = purify_depolarized
            self.purify_grad = purify_depolarized_grad

    def build(self, shape='shallowest') -> Branch:
        """
        shape:
            - 'shallowest' : shallowest tree
            - 'link' : link tree
            - 'rlink' : reverse link tree
            - 'random' : random tree
        """

        def build_shallowest():
            current_nodes: 'list[int]' = list(self.leaves.keys())
            next_nodes: 'list[int]' = []
            # merge nodes layer by layer
            while len(current_nodes) >= 1:
                if len(current_nodes) == 1:
                    self.root = self.nodes[current_nodes[0]]
                    break
                # merge nodes in current layer
                while len(current_nodes) >= 2:
                    id1, id2 = current_nodes.pop(0), current_nodes.pop(0)
                    node1, node2 = self.nodes[id1], self.nodes[id2]
                    f1, f2 = node1.fidelity, node2.fidelity
                    f = self.swap_func(f1, f2)
                    new_id = Node.get_new_id()
                    edge = node1.edge_tuple[0], node2.edge_tuple[1], -1
                    new_node = Branch(new_id, edge, None, node1, node2, f, BranchOp.SWAP)
                    node1.parent = new_node
                    node2.parent = new_node
                    self.nodes[new_id] = new_node
                    next_nodes.append(new_id)

                    # print(f'New Node {new_id} = {node1} + {node2}')
                    # print(f'Fidelity {f} = swap({f1}, {f2})')
                
                if len(current_nodes) == 1:
                    next_nodes.append(current_nodes.pop())
                
                current_nodes, next_nodes = next_nodes, current_nodes

        def build_link(reverse=False):
            current_nodes = list(self.leaves.keys())
            if len(current_nodes) == 1:
                    self.root = self.nodes[current_nodes[0]]
                    return

            while len(current_nodes) >= 1:
                if len(current_nodes) == 1:
                    self.root = self.nodes[current_nodes[0]]
                    return
                if reverse:
                    id1, id2 = current_nodes.pop(), current_nodes.pop()
                else:
                    id1, id2 = current_nodes.pop(0), current_nodes.pop(0)
                node1, node2 = self.nodes[id1], self.nodes[id2]
                f1, f2 = node1.fidelity, node2.fidelity
                f = self.swap_func(f1, f2)
                new_id = Node.get_new_id()
                edge = node1.edge_tuple[0], node2.edge_tuple[1], -1
                new_node = Branch(new_id, edge, None, node1, node2, f, BranchOp.SWAP)
                node1.parent = new_node
                node2.parent = new_node
                self.nodes[new_id] = new_node
                if reverse:
                    current_nodes.append(new_id)
                else:
                    current_nodes.insert(0, new_id)

                # print(f'New Node {new_id} = {node1} + {node2}')
                # print(f'Fidelity {f} = swap({f1}, {f2})')

        def build_random():
            current_nodes = list(self.leaves.keys())
            if len(current_nodes) == 1:
                    self.root = self.nodes[current_nodes[0]]
                    return

            while len(current_nodes) >= 1:
                if len(current_nodes) == 1:
                    self.root = self.nodes[current_nodes[0]]
                    return
                index = np.random.randint(0, len(current_nodes) - 1)
                id1, id2 = current_nodes.pop(index), current_nodes.pop(index)
                node1, node2 = self.nodes[id1], self.nodes[id2]
                f1, f2 = node1.fidelity, node2.fidelity
                f = self.swap_func(f1, f2)
                new_id = Node.get_new_id()
                edge = node1.edge_tuple[0], node2.edge_tuple[1], -1
                new_node = Branch(new_id, edge, None, node1, node2, f, BranchOp.SWAP)
                node1.parent = new_node
                node2.parent = new_node
                self.nodes[new_id] = new_node
                current_nodes.insert(index, new_id)

                # print(f'New Node {new_id} = {node1} + {node2}')
                # print(f'Fidelity {f} = swap({f1}, {f2})')

        
        if shape == 'shallowest':
            build_shallowest()
        elif shape == 'link':
            build_link()
        elif shape == 'rlink':
            build_link(reverse=True)
        elif shape == 'random':
            build_random()
        else:
            raise ValueError(f'Unsopported shape {shape}')
        
        return self.root

    def purify(self, node: Node) -> Branch:
        """
        purify a node in the tree
        return the new node (parent of the given node and its copy)
        """
        parent = node.parent
        copy_node = self.copy_subtree(node)
        f1, f2 = node.fidelity, copy_node.fidelity
        fid = self.purify_func(f1, f2)
        new_id = Node.get_new_id()
        edge = node.edge_tuple[0], copy_node.edge_tuple[1], -1
        new_node = Branch(new_id, edge, parent, copy_node, node, fid, BranchOp.PURIFY)
        self.nodes[new_id] = new_node

        copy_node.parent = new_node
        node.parent = new_node
        if parent is not None:
            if parent.left == node:
                parent.left = new_node
            else:
                parent.right = new_node
        else:
            self.root = new_node

        return new_node
        
    def purify_id(self, node_id: int) -> Branch:
        """
        purify a node by id
        """
        return self.purify(self.nodes[node_id])       

    def grad(self, node: Branch=None, self_grad: float=1) -> None:
        """
        Calculate the grads of all descendants, wrt the given node
        self_grad is the grad of the node itself (got from its parent)

        !!! Purification is not supported yet.
        """

        if node is None:
            node = self.root
        
        if node.parent is None:
            # f = node.fidelity
            # g = (2*f*(f**2 + (1-f)**2) - f**2*(2*f - 2*(1-f))) / ((f*f + (1-f)*(1-f))**2)**2
            node.grad = 1
            self_grad = 1
        
        if node.is_leaf():
            return

        # non-leaf node must have two children
        assert node.left is not None and node.right is not None, 'node must have two children'

            
        # calculate the grads of children
        f1, f2 = node.left.fidelity, node.right.fidelity
        if node.op == BranchOp.SWAP:
            g1 = self.swap_grad(f1, f2, 1) * self_grad
            g2 = self.swap_grad(f1, f2, 2) * self_grad
        elif node.op == BranchOp.PURIFY:
            g1 = self.purify_grad(f1, f2, 1) * self_grad
            g2 = self.purify_grad(f1, f2, 2) * self_grad
        
        # update the grads of children
        node.left.grad = g1
        node.right.grad = g2

        # calculate the grads of children's children
        self.grad(node.left, g1)
        self.grad(node.right, g2)

    def backward(self, node: Branch) -> None:
        """
        update fidelity of all ancestors (not including itself)
        backtrace from node to root
        """
        assert node is not None, 'Cannot backtrace from None'
        # cannot and shouldn't update fidelity of a leaf node
        # assert isinstance(node, Branch), 'Must backtrace from a Branch'

        node = node.parent
        while node is not None:
            # lu, lv, lk = node.left.edge_tuple
            # ru, rv, rk = node.right.edge_tuple
            if node.op == BranchOp.SWAP:
                node.fidelity = self.swap_func(node.left.fidelity, node.right.fidelity)
            elif node.op == BranchOp.PURIFY:
                node.fidelity = self.purify_func(node.left.fidelity, node.right.fidelity)
            node = node.parent
 
    def calc_efficiency(self, node: Node):
        """
        Calculate the efficiency of all descendants, wrt the given node
        efficiency = grad / 2^height
        """
        if node is None:
            return

        # non-leaf node must have two children
        # assert node.left is not None and node.right is not None, 'node must have two children'

        # calculate the efficiency
        node.efficiency = node.grad / SPSTree.count_leaves(node)
        # calculate adjusted efficiency
        f = node.fidelity
        pf = self.purify_func(f, f)
        node.adjust = pf - f
        node.adjust_eff = node.adjust * node.efficiency
        node.test_rank_attr = node.adjust_eff * SPSTree.count_leaves(node)
        
        # calculate the efficiency of descendants
        self.calc_efficiency(node.left)
        self.calc_efficiency(node.right)


class TreeSolver(SPSSolver):
    """
    SPS Solver with a tree structure
    """

    def __init__(self, edges: dict[EdgeTuple, float], system_type) -> None:
        super().__init__(edges)
        assert len(edges) > 0, 'edges must be more than 0'
        
        self.tree = SPSTree(edges, system_type)

    def build(self, shape: str='shallowest') -> None:
        """
        Build the tree
        shape : str, shape of the tree
            - 'shallowest' : shallowest tree
            - 'link' : linked tree
            - 'rlink' : reverse linked tree
            - 'random' : random tree
        """
        self.tree.build(shape)
        self.tree.grad(self.tree.root)
        self.tree.calc_efficiency(self.tree.root)

    def solve(self, fth: float=0.90, attr: str='adjust_eff', report=False) -> tuple[float, int]:
        """
        Parameters
        ----------
        fth : float, root fidelity threshold
        attr : str, attribute used to rank candidate nodes for purifications
            - 'grad' : gradient
            - 'efficiency' : efficiency
            - 'adjust_eff' : adjusted efficiency
        """
        
        sac_epr_num = 0

        while self.tree.root.fidelity < fth:
            # find the node with the highest adjusted efficiency
            node = SPSTree.find_max(self.tree.root, attr=attr)
            if not isinstance(node, Leaf):
                print(f"Intermediate Link Purified: {node.edge_tuple}")
                SPSTree.print_tree(self.tree.root)
                raise
            # purify the node
            sac_epr_num += SPSTree.count_leaves(node)
            node = self.tree.purify(node)
            # update fidelity of all ancestors
            self.tree.backward(node)
            # update the efficiency in the tree
            self.tree.grad(self.tree.root)
            self.tree.calc_efficiency(self.tree.root)

            if report:
                print(f"link purified: {node.edge_tuple}")

            if report and node.edge_tuple[-1] != 0:
                print(f"Intermediate Link Purified: {node.edge_tuple}")
                SPSTree.print_tree(self.tree.root)

        return self.tree.root.fidelity, sac_epr_num

    def check_theorems(self, node: Node, report=False) -> None:
        """
        Confirm the theorems
        """
        
        if node.is_leaf():
            return
        # non-leaf node must have two children
        assert node.left is not None and node.right is not None, 'node must have two children'

        # confirm the theorem of grad
        assert node.grad >= node.left.grad and node.grad >= node.right.grad, \
            'grad theorem is not satisfied at node {}'.format(node.edge_tuple)

        # confirm the theorem of efficiency
        # assert node.efficiency <= node.left.efficiency and node.efficiency <= node.right.efficiency, \
        #     'efficiency theorem is not satisfied at node {}'.format(node.edge_tuple)
        
        # confirm the theorem of adjust
        assert node.adjust <= node.left.adjust and node.adjust <= node.right.adjust, \
            'adjust theorem is not satisfied at node {}'.format(node.edge_tuple)

        # confirm the theorem of adjusted efficiency
        assert node.adjust_eff <= node.left.adjust_eff and node.adjust_eff <= node.right.adjust_eff, \
            'adjusted efficiency theorem is not satisfied at node {}'.format(node.edge_tuple)
        
        if node.is_root() and report:
            print('All theorems are satisfied!')

    def rec_solve(self, fth: float=0.90, attr: str='adjust_eff', report=False) -> tuple[float, int]:
        """
        recursivly solve the tree
        branch and rec_solve at node with adjust_eff >= 0.5
        ----------
        fth : float, root fidelity threshold
        attr : str, attribute used to rank candidate nodes for purifications
            - 'grad' : gradient
            - 'efficiency' : efficiency
            - 'adjust_eff' : adjusted efficiency
        """
        
        sac_epr_num = 0

        while self.tree.root.fidelity < fth:
            # find the node with the highest adjusted efficiency
            node = SPSTree.find_max(self.tree.root, attr=attr)
            # if node.edge_tuple[-1] != 0:
            #     print(f"Intermediate Link Purified: {node.edge_tuple}")
            #     SPSTree.print_tree(self.tree.root)
            #     raise
            # purify the node
            sac_epr_num += SPSTree.count_leaves(node)
            node = self.tree.purify(node)
            # update fidelity of all ancestors
            self.tree.backward(node)
            # update the efficiency in the tree
            self.tree.grad(self.tree.root)
            self.tree.calc_efficiency(self.tree.root)

            if report:
                print(f"link purified: {node.edge_tuple}")

            if report and node.edge_tuple[-1] != 0:
                print(f"Intermediate Link Purified: {node.edge_tuple}")
                SPSTree.print_tree(self.tree.root)

        return self.tree.root.fidelity, sac_epr_num


class EPPSolver(SPSSolver):
    """
    EPP in EFiRAP
    """
    def __init__(self, edges: dict[EdgeTuple, float], system_type=SystemType.DEPHASED) -> None:
        super().__init__(edges)
        assert len(edges) > 1, 'edges must be more than 1'

        self.SST = SPSTree(edges, system_type=system_type)
        self.system_type = system_type

        if system_type == SystemType.DEPHASED:
            self.swap_func = swap_dephased
            self.purify_func = purify_dephased
        elif system_type == SystemType.DEPOLARIZED:
            self.swap_func = swap_depolarized
            self.purify_func = purify_depolarized

        
        # self.edge_tuples = list(edges.keys())
        N = len(edges) - 1
        # Equ. 11
        self.f_min = ((3*N-1)**2 + 1) / (3*N)**2


    def min_purify(self) -> tuple[dict[EdgeTuple, int], dict[EdgeTuple, float]]:
        """
        Purify the edges to f_min
        """
        # # of purification per edge
        edges = copy.deepcopy(self.edges)
        A = { edge_tuple: 0 for edge_tuple in edges.keys() }

        for edge_tuple in edges.keys():
            fl = edges[edge_tuple]
            f = fl
            while f < self.f_min:
                f = self.purify_func(f, fl)
                A[edge_tuple] += 1
            edges[edge_tuple] = f

        # keep non-zero elements
        A = { k: v for k, v in A.items() if v > 0 }
        return A, edges

    def purify_n(self, fc: float, fl: float, n: int) -> None:
        """
        Purify the edges by n times
        """
        assert n > 0, 'n must be more than 0'
        ft = self.purify_func(fc, fl)
        n -= 1
        while n > 0:
            ft = self.purify_func(ft, fl)
            n -= 1
        return ft

    def swap_along(self, fs: 'list[float]') -> float:
        """
        Swap the fidelities of the edges
        """
        assert len(fs) > 1, 'fs must be more than 1'
        f = fs[0]
        for i in range(1, len(fs)):
            f = self.swap_func(f, fs[i])
        return f
    
    def find_critical_link(self, edges: dict[EdgeTuple, float], attr='grad') -> EdgeTuple:
        """
        Find the critical link
        """
        # find the critical link
        # critical link: the link with the largest fradient

        tree = SPSTree(edges, system_type=self.system_type)
        tree.build()
        tree.grad(tree.root)
        tree.calc_efficiency(tree.root)
        # SPSTree.print_tree(tree.root)
        node = tree.find_max(tree.root, attr=attr, search_range=[Leaf])
        return node.edge_tuple

    def solve(self, fth, attr='grad') -> tuple[float, int, int]:
        """
        Solve the problem
        """
        sac_epr_num = 0
        A, temp = self.min_purify()
        min_purify_num = sum(A.values())
        # recording current fidelities of the edges
        edges = copy.deepcopy(self.edges)
        fids = list(edges.values())
        e2e_f = self.swap_along(fids)
        if e2e_f >= fth:
            return e2e_f, sac_epr_num, min_purify_num
        else:
            if len(A) == 0:
                edge_tuple = self.find_critical_link(edges, attr=attr)
                A[edge_tuple] = A[edge_tuple] + 1 if edge_tuple in A else 1

        # recording edges to be purified
        # edge_list = list(A.keys())
        C = {}
        while len(A) > 0:
            edge, num = list(A.items())[0]
            A.pop(edge)
            fc = edges[edge]
            fl = self.edges[edge]
            sac_epr_num += num
            f = self.purify_n(fc, fl, num)
            edges[edge] = f

            fids = list(edges.values())
            e2e_f = self.swap_along(fids)
            if e2e_f >= fth:
                break
            else:
                edge_tuple = self.find_critical_link(edges, attr=attr)
                A[edge_tuple] = A[edge_tuple] + 1 if edge_tuple in A else 1

        return e2e_f, sac_epr_num, min_purify_num


class DPSolver(SPSSolver):
    # dynamic programming solver

    def __init__(self, 
            edges: dict[EdgeTuple, float],
            system_type=SystemType.DEPHASED,
            budget: int = 100,
            ) -> None:
        super().__init__(edges,)
        
        # self.budget = budget
        self.system_type=system_type

        if self.system_type == SystemType.DEPHASED:
            self.swap_func = swap_dephased
            self.swap_grad = swap_dephased_grad
            self.purify_func = purify_dephased
            self.purify_grad = purify_dephased_grad
        else:
            self.swap_func = swap_depolarized
            self.swap_grad = swap_depolarized_grad
            self.purify_func = purify_depolarized
            self.purify_grad = purify_depolarized_grad

        self.edge_num = len(edges)
        self.fids = list(edges.values())
        self.alloc = {}

    def purify(self, fid, budget) -> float:
        # f = fid
        # balanced purify
        fids: list = [fid] * (budget + 1)
        while len(fids) > 1:
            f1 = fids.pop(0)
            f2 = fids.pop(0)
            fids.append(self.purify_func(f1, f2))

        # for l in range(budget):
        #     f = self.purify_func(f, fid)

        return fids[0]
    
    def solve(self, budget: int) -> float:
        self.mat = np.zeros((self.edge_num, self.edge_num+1, budget+1), dtype=np.float64)
        # init dp matrix
        for i in range(self.edge_num):
            for l in range(budget+1):
                self.mat[i][i+1][l] = self.purify(fids[i], l)

        # self.alloc[(0, self.edge_num, 0)] = budget
        # f = self.solve_rec(0, self.edge_num, budget)

        for length in range(2, self.edge_num + 1):
            for i in range(self.edge_num - length + 1):
                j = i + length
                # got all subpaths of length len
                # now try all possible budget for each subpath
                for k in range(budget+1):
                    max_fid = 0
                    # try all possible path split
                    for m in range(i+1, j):
                        # try all possible budget split
                        for b in range(k+1):
                            fid_left = self.mat[i][m][b]
                            fid_right = self.mat[m][j][k-b]
                            fid = self.swap_func(fid_left, fid_right)
                            if fid > max_fid:
                                max_fid = fid
                                max_m = m
                                max_b = b
                                max_k = k

                    self.mat[i][j][k] = max_fid
                    self.alloc[(i, j, k)] = (max_m, max_b, max_k - max_b)
        
        
        def recover_alloc(src_alloc, dst_alloc, i, j, k):
            m, bl, br = src_alloc[(i, j, k)]
            dst_alloc[(i, m)] = bl
            dst_alloc[(m, j)] = br
            if m - i > 1:
                recover_alloc(src_alloc, dst_alloc, i, m, bl)
            if j - m > 1:
                recover_alloc(src_alloc, dst_alloc, m, j, br)
        allocs = {}
        recover_alloc(self.alloc, allocs, 0, self.edge_num, budget)

        alloc = [0] * self.edge_num
        for i in range(self.edge_num):
            alloc[i] = allocs[(i, i+1)]
        
        f = self.mat[0][self.edge_num][budget]
        return f, alloc

    def solve_rec(self, i, j, k) -> float:
        """
        from node i to j, budget k
        """

        # basic case: only one edge
        if j == i + 1:
            return self.purify(self.fids[i], k)


        max_fid = 0
        for m in range(i+1, j):
            for l in range(k+1):
                fid_left = self.solve_rec(i, m, l)
                fid_right = self.solve_rec(m, j, k-l)
                fid = self.swap_func(fid_left, fid_right)
                if fid > max_fid:
                    max_fid = fid
                    self.alloc[(i, m, l)] = fid_left
                    self.alloc[(m, j, k-l)] = fid_right

        self.alloc[(i, j, k)] = max_fid
        return max_fid



class GRDYSolver(SPSSolver):
    # greedy solver
    def __init__(self, 
            edges: dict[EdgeTuple, float],
            system_type=SystemType.DEPHASED,
            fth: float = 0.9,
            ) -> None:
        super().__init__(edges,)
        
        # self.budget = budget
        self.system_type=system_type

        if self.system_type == SystemType.DEPHASED:
            self.swap_func = swap_dephased
            self.swap_grad = swap_dephased_grad
            self.purify_func = purify_dephased
            self.purify_grad = purify_dephased_grad
        else:
            self.swap_func = swap_depolarized
            self.swap_grad = swap_depolarized_grad
            self.purify_func = purify_depolarized
            self.purify_grad = purify_depolarized_grad

        self.edge_num = len(edges)
        self.fids = list(edges.values())
        self.alloc = { edge_tuple: 0 for edge_tuple in edges.keys() }

    def purify(self, fid, budget) -> float:
        # balanced purify
        fids: list = [fid] * (budget + 1)
        while len(fids) > 1:
            f1 = fids.pop(0)
            f2 = fids.pop(0)
            fids.append(self.purify_func(f1, f2))

        return fids[0]

    def swap_purify(self, fids, allocs) -> float:

        pfids = np.zeros(self.edge_num)
        for i in range(self.edge_num):
            pfids[i] = self.purify(fids[i], allocs[i])

        # swap along pfids
        f = 0.5 + pow(2, self.edge_num - 1) * np.prod(pfids - 0.5)

        return f
    
    def solve(self, fth: int) -> float:
        allocs = [0] * self.edge_num
        f = 0
        while f < fth:
            fm = 0
            em = 0
            for i in range(self.edge_num):
                allocs[i] += 1
                f_new = self.swap_purify(self.fids, allocs)
                if f_new > fm:
                    fm = f_new
                    em = i

                allocs[i] -= 1

            allocs[em] += 1
            f = fm

        return f, allocs



def test_IP():
    np.random.seed(0)
    fids = np.random.random(10)
    fids = (fids * 25 + 70) / 100
    # fids = [0.9] * 10
    print(fids)
    edges = {
        (0, 1, 0): fids[0],
        (1, 2, 0): fids[1],
        (2, 3, 0): fids[2],
        # (3, 4, 0): fids[3],
        # (4, 5, 0): fids[4],
        # (5, 6, 0): fids[5],
        # (6, 7, 0): fids[6],
        # (7, 8, 0): fids[7],
        # (8, 9, 0): fids[8],
        # (9, 10, 0): fids[9],
    }
    solver = IPSolver(edges)
    solver.build()
    print(solver.model.getConstrs())
    solver.solve()

    print(solver.A)
    print(solver.M.values())

def test_SPST():
    np.random.seed(0)
    fids = np.random.random(10)
    fids = (fids * 25 + 70) / 100
    # fids = [0.9] * 10
    print(fids)
    edges = {
        (0, 1, 0): fids[0],
        (1, 2, 0): fids[1],
        (2, 3, 0): fids[2],
        (3, 4, 0): fids[3],
        (4, 5, 0): fids[4],
        (5, 6, 0): fids[5],
        (6, 7, 0): fids[6],
        (7, 8, 0): fids[7],
        (8, 9, 0): fids[8],
        (9, 10, 0): fids[9],
    }


    tree = SPSTree(edges)
    tree.build('shallowest')
    # tree.build('link')
    # tree.build('rlink')
    # tree.build('random')
    tree.grad(tree.root)
    tree.calc_efficiency(tree.root)
    SPSTree.print_tree(tree.root)
    # update grads of the whole tree
    # tree.purify_id(0)
    # tree.purify_id(1)
    # node = tree.purify_id(2)
    # tree.backward(node)
    # tree.purify_id(3)
    # tree.purify_id(4)
    # tree.purify_id(5)
    # tree.purify_id(18)

    tree.grad(tree.root)
    tree.calc_efficiency(tree.root)
    SPSTree.print_tree(tree.root)
    SPSTree.traverse(tree.root)

def test_TreeSolver(edges: dict[EdgeTuple, float],
                    fth: float=0.9,
                    shape: str='random',
                    system_type: SystemType=SystemType.DEPHASED,):
    # fidelity of initial root
    # pred_fid = pow(4/3, len(edges)-1)
    # for edge, fid in edges.items():
    #     pred_fid *= (fid - 1/4)
    # pred_fid += 1/4
    # print(pred_fid)

    solver = TreeSolver(edges, system_type)
    # shape = 'shallowest'
    # shape = 'link'
    # shape = 'rlink'
    solver.build(shape=shape)
    # SPSTree.print_tree(solver.tree.root)

    # try:
        # solver.check_theorems(solver.tree.root)
    # except AssertionError as e:
    #     print(e)
    #     SPSTree.print_tree(solver.tree.root)
        # raise e

    # the last 3 may not be used for ranking, may not converge
    attrs=['adjust_eff', 'adjust', 'test_rank_attr', 'fidelity', 'grad', 'efficiency',  ]
    f, num = solver.solve(fth=fth, attr=attrs[0], report=False)
    # SPSTree.print_tree(solver.tree.root)

    print(f, num)
    # print("leaf num: ", SPSTree.count_leaves(solver.tree.root))

def test_EPPSolver(edges: dict[EdgeTuple, float], fth: float=0.9, system_type: SystemType=SystemType.DEPHASED,):
    solver = EPPSolver(edges, system_type)
    f, num, min_num = solver.solve(fth=fth, attr='adjust_eff')
    print(f, num, min_num)

def test_DPSolver(edges: dict[EdgeTuple, float], budget=10, system_type: SystemType=SystemType.DEPHASED,):
    solver = DPSolver(edges, system_type)
    f, allocs = solver.solve(budget=budget)
    print(f)
    print(allocs)

def test_GRDYSolver(edges: dict[EdgeTuple, float], fth: float=0.9, system_type: SystemType=SystemType.DEPHASED,):
    solver = GRDYSolver(edges, system_type)
    f, allocs = solver.solve(fth=fth)
    print(f)
    print(allocs)

if __name__ == "__main__":
    np.random.seed(0)
    fids = np.random.random(100)
    fid_lower_bound = 0.7
    fid_range = 0.25
    fids = fids * fid_range + fid_lower_bound
    # fids = [0.9] * 10
    # print(fids)
    fth=0.9
    edge_num = 10
    edges = {
        (i, i+1, 0): fids[i] for i in range(edge_num)
    }

    np.random.seed(int(time.asctime().split()[-2][-2:]))
    # test_IP()
    # test_SPST()
    # for i in range(100):
    test_TreeSolver(edges, fth, shape='shallowest', system_type=SystemType.DEPHASED)
    # test_TreeSolver(edges, fth, shape='shallowest', system_type=SystemType.DEPOLARIZED)
    test_EPPSolver(edges, fth, system_type=SystemType.DEPHASED)
    # test_EPPSolver(edges, fth, system_type=SystemType.DEPOLARIZED)

    test_DPSolver(edges, 17, system_type=SystemType.DEPHASED)
    # test_DPSolver(edges, 92, system_type=SystemType.DEPOLARIZED)
    test_GRDYSolver(edges, fth, system_type=SystemType.DEPHASED)
    # test_GRDYSolver(edges, fth, system_type=SystemType.DEPOLARIZED)

    print("Done!")
