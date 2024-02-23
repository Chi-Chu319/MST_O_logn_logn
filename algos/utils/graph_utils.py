from typing import List

from algos.cluster_edge import ClusterEdge
from algos.graph import GraphLocal


class GraphUtil:
    @staticmethod
    def get_min_weight_to_cluster_edges(graph_local: GraphLocal, cluster_leaders: List[int]) -> List[List[ClusterEdge]]:
        # Compute the minimum-weight edge e(v, F') that connects v to (any node of) F' for all clusters F' not = F.
        vertex_local_start = graph_local.get_vertex_local_start()
        comm_size = graph_local.get_comm_size()
        vertices = graph_local.get_vertices()

        sendbuf = [[] for _ in range(comm_size)]

        for vertex_local, edges in enumerate(vertices):
            vertex = vertex_local + vertex_local_start

            cluster_edges = [
                ClusterEdge(
                    from_cluster=cluster_leaders[vertex],
                    from_v=vertex,
                    to_cluster=cluster_leaders[edge.get_to()],
                    edge=edge.copy()
                )
                for edge in edges if not edge.chosen
            ]

            min_cluster_edges = {}

            for cluster_edge in cluster_edges:
                to_cluster = cluster_edge.get_to_cluster()
                if to_cluster not in min_cluster_edges or min_cluster_edges[to_cluster].edge.get_weight() > cluster_edge.edge.get_weight():
                    min_cluster_edges[to_cluster] = cluster_edge

            for cluster_edge in min_cluster_edges.values():
                cluster_leader = cluster_edge.get_to_cluster()
                sendbuf[graph_local.get_vertex_machine(cluster_leader)].append(cluster_edge)

        return sendbuf

    @staticmethod
    def get_min_weight_from_cluster_edges(cluster_edges: List[ClusterEdge]) -> List[ClusterEdge]:
        """Filter the edges with the same from_cluster/to_v and get the minimum weight edge from each cluster"""

        min_cluster_edges = {}

        for cluster_edge in cluster_edges:
            from_cluster = cluster_edge.get_from_cluster()
            if from_cluster not in min_cluster_edges or min_cluster_edges[from_cluster].edge.get_weight() > cluster_edge.edge.get_weight():
                min_cluster_edges[from_cluster] = cluster_edge

        cluster_edges_unique_from_cluster = list(min_cluster_edges.values())

        min_cluster_edges = {}

        for cluster_edge in cluster_edges_unique_from_cluster:
            to_v = cluster_edge.get_edge().get_to()
            if to_v not in min_cluster_edges or min_cluster_edges[to_v].edge.get_weight() > cluster_edge.edge.get_weight():
                min_cluster_edges[to_v] = cluster_edge

        result = list(min_cluster_edges.values())

        return result
