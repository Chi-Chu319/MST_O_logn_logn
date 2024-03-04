from typing import List

import numpy as np
import sys
from numpy import ndarray


class DistGraphLocal:
    def __init__(
            self,
            comm_size: int,
            rank: int,
            num_vertex_local: int,
            max_weight: int,
    ):
        self.rng = np.random.default_rng()
        self.rank = rank
        self.max_weight = max_weight
        self.comm_size = comm_size
        self.num_vertex_local = num_vertex_local
        self.num_vertices = self.comm_size * self.num_vertex_local
        self.vertex_local_start = rank * num_vertex_local
        self.vertices = np.zeros((self.num_vertex_local, self.num_vertices))

    def __random_weight(self) -> float:
        return self.rng.random() * self.max_weight

    def set_vertices(self, vertices: ndarray):
        self.vertices = vertices

    def generate(self):
        sendbuf = [[] for _ in range(self.comm_size)]

        for vertex_local in range(self.num_vertex_local):
            vertex_from = vertex_local + self.vertex_local_start
            for vertex_to in range(0, vertex_from):
                random_weight = self.__random_weight()
                self.vertices[vertex_local][vertex_to] = random_weight
                sendbuf[self.get_vertex_machine(vertex_to)].append(random_weight)

        return sendbuf

    def fill(self, recvbuf: List[List[float]]):
        for from_rank, edges in enumerate(recvbuf):
            idx = 0
            if from_rank < self.rank:
                continue
            for vertex_local in range(self.num_vertex_local):
                vertex_from = vertex_local + self.num_vertex_local * from_rank
                for vertex_to in range(self.vertex_local_start, min(vertex_from, self.vertex_local_start + self.num_vertex_local)):
                    self.vertices[vertex_to - self.vertex_local_start][vertex_from] = edges[idx]
                    idx += 1

        for vertex_local in range(self.num_vertex_local):
            vertex = vertex_local + self.vertex_local_start
            self.vertices[vertex_local][vertex] = sys.maxsize

    def get_vertex_local_start(self) -> int:
        return self.vertex_local_start

    def get_num_vertex_local(self) -> int:
        return self.num_vertex_local

    def get_vertices(self) -> ndarray:
        return self.vertices

    def get_comm_size(self) -> int:
        return self.comm_size

    def get_vertex_machine(self, vertex: int) -> int:
        return vertex // self.num_vertex_local

    def __str__(self) -> str:
        result = f'rank: {self.rank}\n'

        for i, edges in enumerate(self.vertices):
            result += '\n'
            for edge in edges:
                result += f'{str(edge)} '

        return result

class Graph:
    def __init__(self, comm_size: int, num_vertex_local: int, expected_degree: int, max_weight: int,
                 is_clique: bool) -> None:
        self.rng = np.random.default_rng(18)

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

    def split(self) -> List[DistGraphLocal]:
        result = []

        for i in range(self.comm_size):
            graph_local = DistGraphLocal(
                rank=i,
                comm_size=self.comm_size,
                num_vertex_local=self.num_vertex_local,
                max_weight=self.max_weight,
            )
            graph_local.set_vertices(self.vertices[i * self.num_vertex_local: (i + 1) * self.num_vertex_local])

            result.append(graph_local)

        return result

    def __str__(self) -> str:
        result = ""

        for i, edges in enumerate(self.vertices):
            result += '\n'
            for edge in edges:
                result += f'{str(edge)} '

        return result
