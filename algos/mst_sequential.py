from algos.graph import Graph
import sys


def __min_weight(graph, min_weights, in_mst):
    """Min weight vertex to the current MST, for any v not in the MST"""
    min_weight = sys.maxsize

    min_index = -1

    for v in range(graph.num_vertices):
        if min_weights[v] < min_weight and not in_mst[v]:
            min_weight = min_weights[v]
            min_index = v

    return min_index


def mst_sequential(graph: Graph):
    # v -> the lowest weight to reach the current MST (v is not in the MST) if v is in the MST then shortest[v] = 0
    min_weights = [sys.maxsize] * graph.num_vertices
    parent = [-1] * graph.num_vertices
    min_weights[0] = 0
    in_mst = [False] * graph.num_vertices
    parent[0] = -1  # root
    vertices = graph.vertices

    for _ in range(graph.num_vertices):
        u = __min_weight(graph, min_weights, in_mst)
        in_mst[u] = True

        # The vertex is not connected to itself
        for edge in vertices[u]:
            v = edge.get_to()
            if 0 < edge.get_weight() < min_weights[v] and not in_mst[v]:
                min_weights[v] = edge.get_weight()
                parent[v] = u

    return parent
