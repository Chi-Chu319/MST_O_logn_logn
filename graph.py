from typing import List

import numpy as np
from weighted_edge import WeightedEdge


class GraphLocal:
    def __init__(
            self,
            comm_size: int,
            rank: int,
            num_vertex_local: int,
            vertices: List[List[WeightedEdge]]
    ):
        self.rank = rank
        self.comm_size = comm_size
        self.num_vertex_local = num_vertex_local
        self.num_vertices = self.comm_size * self.num_vertex_local
        self.vertices = vertices
        self.vertex_local_start = rank * num_vertex_local

    def __str__(self) -> str:
        result = f'rank: {self.rank}\n'

        for i, edges in enumerate(self.vertices):
            result += f'Vertex: {i + self.vertex_local_start}\n'
            for edge in edges:
                result += f'{str(edge)}\n'

        return result


class Graph:
    def __init__(self, comm_size: int, num_vertex_local: int, expected_degree: int, max_weights: int) -> None:
        self.rng = np.random.default_rng()

        self.comm_size = comm_size
        self.num_vertex_local = num_vertex_local
        self.expected_degree = expected_degree
        self.max_weights = max_weights
        self.num_vertices = self.comm_size * self.num_vertex_local

        self.vertices = [[] for _ in range(self.num_vertices)]

    def generate(self):
        p = self.expected_degree / (self.num_vertices - 1)

        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if self.rng.random() < p:
                    vertex_i, vertex_j = self.__generate_edges(i, j)
                    self.vertices[i].append(vertex_i)
                    self.vertices[j].append(vertex_j)

    def __generate_edges(self, i: int, j: int) -> (WeightedEdge, WeightedEdge):
        weight = self.rng.random() * self.max_weights
        return WeightedEdge(j, weight), WeightedEdge(i, weight)

    def split(self) -> List[GraphLocal]:
        result = []

        for i in range(self.comm_size):
            result.append(GraphLocal(
                rank=i,
                comm_size=self.comm_size,
                num_vertex_local=self.num_vertex_local,
                vertices=self.vertices[i * self.num_vertex_local: (i + 1) * self.num_vertex_local]
            ))

        return result

    def __str__(self) -> str:
        result = ""

        for i, edges in enumerate(self.vertices):
            result += f'Vertex: {i}\n'
            for edge in edges:
                result += f'{str(edge)}\n'

        return result
