from typing import List

import numpy as np
import sys


class GraphLocal:
    def __init__(
            self,
            comm_size: int,
            rank: int,
            num_vertex_local: int,
            vertices: List[List[float]]
    ):
        self.rank = rank
        self.comm_size = comm_size
        self.num_vertex_local = num_vertex_local
        self.num_vertices = self.comm_size * self.num_vertex_local
        self.vertices = vertices
        self.vertex_local_start = rank * num_vertex_local

    def get_vertex_local_start(self) -> int:
        return self.vertex_local_start

    def get_num_vertex_local(self) -> int:
        return self.num_vertex_local

    def get_vertices(self) -> List[List[float]]:
        return self.vertices

    def get_comm_size(self) -> int:
        return self.comm_size

    def get_vertex_machine(self, vertex: int) -> int:
        return vertex // self.num_vertex_local

    def __str__(self) -> str:
        result = f'rank: {self.rank}\n'

        for i, edges in enumerate(self.vertices):
            result += f'Vertex: {i + self.vertex_local_start}\n'
            for edge in edges:
                result += f'{str(edge)}\n'

        return result


class Graph:
    def __init__(self, comm_size: int, num_vertex_local: int, expected_degree: int, max_weight: int, is_clique: bool) -> None:
        self.rng = np.random.default_rng()

        self.comm_size = comm_size
        self.num_vertex_local = num_vertex_local
        self.expected_degree = expected_degree
        self.max_weight = max_weight
        self.num_vertices = self.comm_size * self.num_vertex_local
        self.vertices = np.zeros((self.num_vertices, self.num_vertices))
        self.clique = is_clique

    def generate(self) -> 'Graph':
        p = self.expected_degree / (self.num_vertices - 1)

        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if (self.rng.random() < p) or self.clique:
                    weight = self.__random_weight()
                    self.vertices[i][j] = weight
                    self.vertices[j][i] = weight

        for i in range(self.num_vertices):
            self.vertices[i][i] = sys.maxsize

        return self
        
    def __random_weight(self) -> float:
        return self.rng.random() * self.max_weight

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
