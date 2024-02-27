

import gurobipy as gp
from gurobipy import GRB



if __name__ == '__main__':
    model = gp.Model("G")
    # Create variables
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    vertices = [0, 1, 2, 3, 4]
    weights = [1, 3, 1, 3, 1, 4, 2]

    # find the shortest path from 0 to 4
    # p[u, v] = 1 if edge (u, v) is on the shortest path from 0 to 4
    p = model.addVars(edges, vtype=GRB.BINARY, name="p")
    # 
