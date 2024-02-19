import numpy as np
from weighted_edge import WeightedEdge


class Graph:
    def __init__(self, comm_size, num_vertex_local, expected_degree, max_weights) -> None:
        self.rng = np.random.default_rng()

        self.comm_size = comm_size
        self.num_vertex_local = num_vertex_local
        self.expected_degree = expected_degree
        self.max_weights = max_weights
        self.num_vertices = self.comm_size * self.num_vertex_local

        self.vertices = [[] for _ in range(self.num_vertices)]

    def generate(self):
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if i != j and self.rng.random() < (self.expected_degree / 2) / (self.num_vertices - 1):
                    vertex_i, vertex_j = self.__generate_edges(i, j)
                    self.vertices[i].append(vertex_i)
                    self.vertices[j].append(vertex_j)

    def __generate_edges(self, i, j):
        weight = self.rng.random() * self.max_weights
        return WeightedEdge(j, weight), WeightedEdge(i, weight)

    def get_vertices(self):
        count = [len(vertex) for vertex in self.vertices]

        idx = 0
        displacement = []
        for degree in count:
            displacement.append(idx)
            idx += degree

        return self.vertices, count, displacement


class GraphLocal:


# class GraphGenerator:
#     def __init__(self, comm_size, comm_rank, num_vertex_local, expected_degree, max_weights) -> None:
#         self.rng = np.random.default_rng()
#
#         self.comm_size = comm_size
#         self.comm_rank = comm_rank
#         self.num_vertex_local = num_vertex_local
#         self.expected_degree = expected_degree
#         self.max_weights = max_weights
#         self.vertex_local_start = self.comm_rank * self.num_vertex_local
#         self.num_vertices = self.comm_size * self.num_vertex_local
#
#     def __get_edges(self):
#         edges = []
#
#         for vertex_local in range(self.num_vertex_local):
#             vertex_edges = np.random.binomial(1, self.expected_degree / (self.num_vertices - 1), self.num_vertices - 1)
#             vertex_edges = np.where(vertex_edges == 1)[0]
#
#             vertex_edges += np.where(vertex_edges >= vertex_local + self.vertex_local_start, 1, 0)
#
#             edges.append(vertex_edges)
#
#         return edges
#
#     def __get_edge_weights(self, edges):
#         weights = []
#
#         for vertex_edges in edges:
#             weights.append(self.rng.random(vertex_edges.shape) * self.max_weights)
#
#         return weights
#
#     def __init_chosen_edges(self, edges):
#         chosen_edges = []
#
#         for vertex_edges in edges:
#             chosen_edges.append(np.zeros(vertex_edges.shape))
#
#         return chosen_edges
#
#     def get_vertex_machine(self, vertex):
#         return vertex % self.num_vertex_local
#
#     def sort(self, edges, weights, chosen_edges):
#         """
#         Sort the edges so that the min weight one comes at first
#         """
#         for vertex_local in range(self.num_vertex_local):
#             sorted_vertex_weights = []
#             sorted_vertex_edges = []
#             sorted_vertex_chosen_edges = []
#
#             for weight, edge, chosen_edge in sorted(
#                     zip(weights[vertex_local], edges[vertex_local], chosen_edges[vertex_local])):
#                 sorted_vertex_weights.append(weight)
#                 sorted_vertex_edges.append(edge)
#                 sorted_vertex_chosen_edges.append(chosen_edge)
#
#             weights[vertex_local] = sorted_vertex_weights
#             edges[vertex_local] = sorted_vertex_edges
#             chosen_edges[vertex_local] = sorted_vertex_chosen_edges
#
#     def generate(self):
#         edges = self.__get_edges()
#         weights = self.__get_edge_weights(edges)
#         chosen_edges = self.__init_chosen_edges(edges)
#
#         self.sort(edges=edges, weights=weights, chosen_edges=chosen_edges)
#
#         return edges, weights, chosen_edges
