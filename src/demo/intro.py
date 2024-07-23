

import networkx as nx


if __name__ == "__main__":

    U = set([1, 2, 3, 4, 5])
    R = set([6, 7, 8, 9, 10])

    V = U + R
    E = [(1, 6), (6, 7), (7, 3) (3, 5), (4, 5), (1, 3)]

    graph = nx.Graph()
    graph.add_nodes_from(V)
    graph.add_edges_from(E)

    nx.draw(graph)
