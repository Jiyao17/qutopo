
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
        costs = [d_cost] * deepest_num + [sd_cost] * second_deepest_num
    else:
        costs = [sd_cost] * second_deepest_num + [d_cost] * deepest_num
    
    
    return costs


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
    
    return channel_capacity


def get_edge_length(capacity: float, photon_rate: float, fiber_loss: float):
    """
    get suggested edge length
    """
    # length = -10 * (1/fiber_loss) * 2 * np.log10(np.sqrt(capacity / photon_rate))
    # simplify the formula
    length = -10 * np.log10(capacity / photon_rate) / fiber_loss

    return length



if __name__ == '__main__':
    costs = [1,] * 13
    swap_prob = 0.5

    costs = complete_swap(costs, swap_prob)
    print(costs)
    costs = sequential_swap(costs, swap_prob)
    print(costs)
    costs = relaxed_complete_swap(costs, swap_prob)
    print(costs)

    # cap = get_edge_capacity(100, 1e4, 0.2)
    # print(cap)
    # length = get_edge_length(100, 1e4, 0.2)
    # print(length)