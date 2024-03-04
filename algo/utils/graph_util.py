from typing import List
from mpi4py import MPI

from algo.graph import Graph, DistGraphLocal
from algo.quick_union import QuickUnionUF

class GraphUtil:
    @staticmethod
    def generate_clique_graph(rank: int, comm_size: int, max_weight: int, num_vertex_local: int):
        if rank == 0:
            return Graph(
                comm_size=comm_size,
                expected_degree=-1,
                max_weight=max_weight,
                num_vertex_local=num_vertex_local,
                is_clique=True
            ).generate()
        else:
            return None

    @staticmethod
    def generate_distribute_clique_graph(comm: MPI.Intracomm, rank: int, comm_size: int, max_weight: int, num_vertex_local: int):
        dist_graph = DistGraphLocal(
            comm_size=comm_size,
            rank=rank,
            max_weight=max_weight,
            num_vertex_local=num_vertex_local,
        )

        sendbuf = dist_graph.generate()
        recvbuf = comm.alltoall(sendbuf)

        dist_graph.fill(recvbuf)

        return dist_graph

    @staticmethod
    def get_min_weight_to_cluster_edges(graph_local: DistGraphLocal, cluster_finder: QuickUnionUF) -> List[List[tuple]]:
        # Compute the minimum-weight edge e(v, F') that connects v to (any node of) F' for all clusters F' not = F.
        vertex_local_start = graph_local.get_vertex_local_start()
        comm_size = graph_local.get_comm_size()
        vertices = graph_local.get_vertices()
        sendbuf = [[] for _ in range(comm_size)]

        min_cluster_edges = []

        for vertex_from_local, edges in enumerate(vertices):
            vertex_from = vertex_from_local + vertex_local_start
            # TODO Dict is more efficient than list (Why?)
            cluster_edges = {}
            for vertex_to, weight in enumerate(edges):
                cluster_from = cluster_finder.get_cluster_leader(vertex_from)
                cluster_to = cluster_finder.get_cluster_leader(vertex_to)
                if cluster_from != cluster_to and ((cluster_to not in cluster_edges) or (cluster_edges[cluster_to][2] > weight)):
                    cluster_edges[cluster_to] = (
                            vertex_from,
                            vertex_to,
                            weight
                        )

            cluster_edges = list(cluster_edges.values())

            min_cluster_edges.extend(cluster_edges)

        for edge in min_cluster_edges:
            sendbuf[graph_local.get_vertex_machine(cluster_finder.get_cluster_leader(edge[1]))].append(edge)

        return sendbuf

    @staticmethod
    def get_min_weight_from_cluster_edges(cluster_edges: List[tuple], cluster_finder: QuickUnionUF) -> List[tuple]:
        """Filter the edges with the same from_cluster/to_v and get the minimum weight edge from each cluster"""

        min_cluster_edges = {}

        for cluster_edge in cluster_edges:
            from_cluster = cluster_finder.get_cluster_leader(cluster_edge[0])
            if from_cluster not in min_cluster_edges or min_cluster_edges[from_cluster][2] > cluster_edge[2]:
                min_cluster_edges[from_cluster] = cluster_edge

        result = list(min_cluster_edges.values())

        return result
