import itertools
from itertools import islice

import matplotlib.pyplot as plt
import networkx as nx

from data.CNFGen import SAT_3
from data.k_sat import KSATVariables


def draw_interaction_graph(var_count: int, clauses: list):
    """ Implements visualization of interactions graphs according to http://www.carstensinz.de/papers/SAT-2005.pdf .
    """
    graph = nx.Graph()
    graph.add_nodes_from([(x, {"color": "red"}) for x in range(var_count)])

    for clause in clauses:
        for u, v in itertools.combinations(clause, 2):
            v_p = abs(v) - 1
            u_p = abs(u) - 1

            if graph.has_edge(v_p, u_p):
                graph[v_p][u_p]["count"] += 1
            else:
                graph.add_edge(abs(v) - 1, abs(u) - 1, count=1)

    edges = graph.edges
    node_color = ["green" for _ in graph]
    edge_width = [graph[u][v]['count'] for u, v in edges]

    options = {
        "edgelist": edges,
        "edge_color": edge_width,
        "node_color": node_color,
        "node_size": 20,
        "width": 1.5,
        "edge_cmap": plt.cm.Greys
    }

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, **options)
    plt.show()


def draw_factor_graph(var_count: int, clauses: list):
    """ We draw variables to clauses bipartite graph.
    Red edge - negative literal. Blue edge - positive literal.
    Green nodes represents clauses and purple clauses represents variables.
    """
    clauses_count = len(clauses)
    graph = nx.Graph()
    graph.add_nodes_from([(x, {"color": "green"}) for x in range(var_count)])
    graph.add_nodes_from([(x, {"color": "black"}) for x in range(var_count, var_count + clauses_count, 1)])

    edges = [(abs(l - 1), idx, "b" if l > 0 else "r") for idx, c in enumerate(clauses, var_count) for l in c]
    graph.add_weighted_edges_from(edges, "color")

    edges = graph.edges
    edge_color = [graph[u][v]['color'] for u, v in edges]
    node_color = ["cyan" if node < var_count else "green" for node in graph]

    options = {
        "edge_color": edge_color,
        "node_color": node_color,
        "node_size": 20,
    }

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, **options)
    plt.show()


def main():
    dataset = SAT_3("/tmp")
    # dataset = KSATVariables("/tmp")
    # dataset = DomSet("/tmp")
    var_count, clauses = [x for x in islice(dataset.train_generator(), 1)][0]

    draw_interaction_graph(var_count, clauses)
    draw_factor_graph(var_count, clauses)


if __name__ == '__main__':
    main()
