from typing import List

from algo.cluster_edge import ClusterEdge
from algo.graph import GraphLocal, Graph
from algo.quick_union import QuickUnionUF


class GraphUtil:
    @staticmethod
    def generate_graph(rank: int, comm_size: int, expected_degree: int, max_weight: int, num_vertex_local: int):
        if rank == 0:
            return Graph(
                comm_size=comm_size,
                expected_degree=expected_degree,
                max_weight=max_weight,
                num_vertex_local=num_vertex_local,
                is_clique=False
            ).generate()
        else:
            return None

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
    def get_min_weight_to_cluster_edges(graph_local: GraphLocal, cluster_finder: QuickUnionUF, initial: bool) -> List[List[ClusterEdge]]:
        # Compute the minimum-weight edge e(v, F') that connects v to (any node of) F' for all clusters F' not = F.
        vertex_local_start = graph_local.get_vertex_local_start()
        comm_size = graph_local.get_comm_size()
        vertices = graph_local.get_vertices()
        sendbuf = [[] for _ in range(comm_size)]

        # if initial:
        #     for vertex_from_local, edges in enumerate(vertices):
        #         vertex_from = vertex_from_local + vertex_local_start
        #         for vertex_to, weight in enumerate(edges):
        #                 cluster_from = cluster_finder.get_cluster_leader(vertex_from)
        #                 cluster_to = cluster_finder.get_cluster_leader(vertex_to)
        #                 sendbuf[graph_local.get_vertex_machine(cluster_to)].append(
        #                     ClusterEdge(
        #                         from_v=vertex_from,
        #                         to_cluster=cluster_to,
        #                         weight=weight,
        #                         to_v=vertex_to,
        #                         from_cluster=cluster_from
        #                     )
        #                 )

        #     return sendbuf
        
        min_cluster_edges = []

        for vertex_from_local, edges in enumerate(vertices):
            vertex_from = vertex_from_local + vertex_local_start
            cluster_edges = [None] * graph_local.num_vertices
            for vertex_to, weight in enumerate(edges):
                cluster_from = cluster_finder.get_cluster_leader(vertex_from)
                cluster_to = cluster_finder.get_cluster_leader(vertex_to)
                if cluster_from != cluster_to and ((cluster_edges[cluster_to] is None) or (cluster_edges[cluster_to].get_weight() > weight)):
                    cluster_edges[cluster_to] = ClusterEdge(
                            from_v=vertex_from,
                            to_cluster=cluster_to,
                            weight=weight,
                            to_v=vertex_to,
                            from_cluster=cluster_from
                        )

            cluster_edges = list(filter(lambda edge: edge is not None, cluster_edges))

            min_cluster_edges.extend(cluster_edges)

        for edge in min_cluster_edges:
            sendbuf[graph_local.get_vertex_machine(edge.get_to_cluster())].append(edge)

        return sendbuf

    @staticmethod
    def get_min_weight_from_cluster_edges(cluster_edges: List[ClusterEdge]) -> List[ClusterEdge]:
        """Filter the edges with the same from_cluster/to_v and get the minimum weight edge from each cluster"""

        min_cluster_edges = {}

        for cluster_edge in cluster_edges:
            from_cluster = cluster_edge.get_from_cluster()
            if from_cluster not in min_cluster_edges or min_cluster_edges[from_cluster].get_weight() > cluster_edge.get_weight():
                min_cluster_edges[from_cluster] = cluster_edge

        cluster_edges_unique_from_cluster = list(min_cluster_edges.values())

        min_cluster_edges = {}

        for cluster_edge in cluster_edges_unique_from_cluster:
            to_v = cluster_edge.get_to_v()
            if to_v not in min_cluster_edges or min_cluster_edges[to_v].get_weight() > cluster_edge.get_weight():
                min_cluster_edges[to_v] = cluster_edge

        result = list(min_cluster_edges.values())

        return result
