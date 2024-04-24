
import numpy as np

def complete_swap(costs: 'list[float]', swap_prob):
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


if __name__ == '__main__':
    costs = [1,] * 7
    swap_prob = 0.7

    cost = complete_swap(costs, swap_prob)
    
    print(cost)