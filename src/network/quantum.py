
import numpy as np


HWParam = {
    'swap_prob': 0.75, # swap probability
    'fiber_loss': 0.2, # fiber loss
    'photon_rate': 1e4, # photon rate
    'pm': 1, # memory price per slot
    # 'pm_install': 1e6, # memory installation price
    'pm_install': 1e1, # memory installation price
    'pc': 1, # channel price per km
    # 'pc_install': 1e4, # channel installation price
    'pc_install': 1e1, # channel installation price
}

def complete_swap(costs: 'list[float]', swap_prob: float):
    """
    conduct swaps as a complete binary tree
    each edge is a leaf
    each swapping is a branching node
    """
    # tree configure
    leaf_num = len(costs)
    tree_depth = int(np.ceil(np.log2(leaf_num)))

    # leaf numbers in the second deepest and deepest level
    second_deepest_num = 2 ** tree_depth - leaf_num
    deepest_num = leaf_num - second_deepest_num

    # single leaf cost contribution to the total cost
    sd_cost = 1/swap_prob**(tree_depth-1)
    d_cost = 1/swap_prob**tree_depth

    # costs array of all nodes
    costs = [d_cost] * deepest_num + [sd_cost] * second_deepest_num
    
    return costs

def relaxed_complete_swap(costs: 'list[float]', swap_prob: float):
    """
    conduct swaps as a  relaxed complete swapping tree
    each edge is a leaf
    each swapping is a branching node
    """
    # tree configure
    leaf_num = len(costs)
    tree_depth = int(np.ceil(np.log2(leaf_num)))

    # leaf numbers in the second deepest and deepest level
    second_deepest_num = 2 ** tree_depth - leaf_num
    deepest_num = leaf_num - second_deepest_num

    # single leaf cost contribution to the total cost
    sd_cost = 1/swap_prob**(tree_depth-1)
    d_cost = 1/swap_prob**tree_depth

    # costs array of all nodes
    # arrange deepest nodes from left or right randomly
    random_num = np.random.rand()
    if random_num < 0.5:
        leaf_costs = [d_cost] * deepest_num + [sd_cost] * second_deepest_num
    else:
        leaf_costs = [sd_cost] * second_deepest_num + [d_cost] * deepest_num
    
    
    return leaf_costs

def sequential_swap(costs: 'list[float]', swap_prob: float):
    """
    conduct swaps as a sequential binary tree
    each edge is a leaf
    each swapping is a branching node
    """
    # tree configure
    leaf_num = len(costs)
    depth = leaf_num - 1

    costs: np.ndarray = np.ones(leaf_num)
    for i in range(1, depth + 1):
        costs[:i + 1] /= swap_prob

    costs = costs.tolist()
    return costs

def get_edge_capacity(length: float, photon_rate: float, fiber_loss: float):
    # prob for half fiber (detectors are in the middle of edges)
    prob = 10 ** (-0.1 * fiber_loss * (length/2)) 
    channel_capacity = photon_rate * prob**2

    # sided light sources and detectors
    # prob = 10 ** (-0.1 * fiber_loss * length) 
    # channel_capacity = photon_rate * prob

    
    return channel_capacity

def get_edge_length(capacity: float, photon_rate: float, fiber_loss: float):
    """
    get suggested edge length
    """
    # length = -10 * (1/fiber_loss) * 2 * np.log10(np.sqrt(capacity / photon_rate))
    # simplify the formula
    length = -10 * np.log10(capacity / photon_rate) / fiber_loss

    return length

def dp_swap(costs: 'list[float]', swap_probs: 'list[float]') -> float:
    """
    solve the swap problem with dynamic programming
    costs: list of costs for each edge
    swap_probs: list of swap probabilities at each node
    """
    # leaf number and node number of the path
    # edge i is between node i and i + 1
    # at node i, the swap probability is swap_probs[i - 1]
    leaf_num = len(costs)
    node_num = leaf_num + 1

    # dp table: mat[i][j] is the cost of the path from i to j
    mat = np.zeros((node_num, node_num))
    # record the split points
    split = {}
    # initialize the leaf costs
    for i in range(leaf_num):
        mat[i][i + 1] = costs[i]
    # dp formula: mat[i][j] = min((mat[i][k] + mat[k][j]) / swap_probs[k - 1] for k in range(i + 1, j))
    for length in range(2, leaf_num + 1):  # length of the path
        for i in range(leaf_num - length + 1):
            j = i + length
            min_cost = float('inf')
            for k in range(i + 1, j):  # split point
                cost = (mat[i][k] + mat[k][j]) / swap_probs[k - 1]
                if cost < min_cost:
                    min_cost = cost
                    split[(i, j)] = k
            mat[i][j] = min_cost
    # the cost of the whole path is in mat[0][leaf_num]
    # return the cost of the whole path
    total_cost = mat[0][leaf_num]

    # calculate the cost of each edge
    edge_costs = [0] * leaf_num
    used_split = {}
    def get_cost(i, j, k, multiplier=1):
        """
        known the cost of the path from i to j is frac_cost,
        return the cost of each edge in the path
        by recursively calculating the cost of shorter paths
        because cost of (i, j) = (cost of (i, k) + cost of (k, j)) / swap_probs[k - 1]
        we reverse the process by maintaining the multiplier = multiplier / swap_probs[k - 1]
        when we reach the leaf edge, we set the edge cost to multiplier
        this cost is the cost of the edge (i, i + 1) when we need 1 E2E entanglement over the path (i, j)
        """
        assert i < j, "i must be less than j"
        if i + 1 == j:  # leaf edge
            edge_costs[i] = multiplier
            return
        split_point = split[(i, j)]
        used_split[(i, j)] = split_point
        # calculate the cost of the left and right paths
        get_cost(i, split_point, k, multiplier / swap_probs[split_point - 1])
        get_cost(split_point, j, k, multiplier / swap_probs[split_point - 1])
        
    # start from the whole path
    get_cost(0, leaf_num, 0, 1)
    # return the total cost and the edge costs
    return total_cost, edge_costs, used_split

def show_splits(
        path_frac: 'tuple[int, int]', 
        splits: 'dict[tuple[int, int], int]', 
        indent: int = 0,
        costs: 'list[float]' = None,
        probs: 'list[float]' = None
        ) -> None:
    """
    show the splits of the path
    """
    start, end = path_frac
    if start == end - 1:
        if costs is not None:
            path_cost = costs[start]
            print('-' * indent + f'{path_cost:.2f}')
        else:
            print('-' * indent)
        return

    mid = splits[path_frac]
    if probs is not None:
        path_prob = probs[mid - 1]
        # assert path_prob >= max(probs[start:end - 1]), f"{path_prob} < {max(probs[start:end - 1])}"
        print('-' * indent + f'{path_frac}, {mid}:{path_prob:.4f}')
    else:
        print('-' * indent + f'{path_frac}')

    show_splits((start, mid), splits, indent + 1, costs, probs)
    show_splits((mid, end), splits, indent + 1, costs, probs)

if __name__ == '__main__':
    # set random seed for reproducibility
    # np.random.seed(42)

    edge_num = 9
    edge_costs = [1,] * edge_num
    base_prob = 0.5
    swap_probs = np.random.rand(edge_num - 1) * (1 - base_prob) + base_prob
    # swap_probs = sorted(swap_probs)

    print(edge_costs)
    print(swap_probs)
    

    # costs = complete_swap(edge_costs, swap_probs[0])
    # print(sum(costs))
    # costs = sequential_swap(edge_costs, swap_probs[0])
    # print(sum(costs))
    # costs = relaxed_complete_swap(edge_costs, swap_probs[0])
    # print(sum(costs))
    cost, edge_costs, split = dp_swap(edge_costs, swap_probs)
    print(cost, sum(edge_costs))
    print(split)

    show_splits((0, edge_num), split, 0, edge_costs, swap_probs)
    
    # always split at larget swap probability

    # cap = get_edge_capacity(100, 1e4, 0.2)
    # print(cap)
    # length = get_edge_length(1, 1e4, 0.2)
    # print(length)