from typing import List

from algo.cluster_edge import ClusterEdge
from algo.graph import GraphLocal
from algo.quick_union import QuickUnionUF


class GraphUtil:
    @staticmethod
    def get_min_weight_to_cluster_edges(graph_local: GraphLocal, cluster_finder: QuickUnionUF) -> List[List[ClusterEdge]]:
        # Compute the minimum-weight edge e(v, F') that connects v to (any node of) F' for all clusters F' not = F.
        vertex_local_start = graph_local.get_vertex_local_start()
        comm_size = graph_local.get_comm_size()
        vertices = graph_local.get_vertices()

        sendbuf = [[] for _ in range(comm_size)]

        for vertex_from_local, edges in enumerate(vertices):
            vertex_from = vertex_from_local + vertex_local_start
            cluster_edges = []
            for vertex_to, weight in enumerate(edges):
                cluster_from = cluster_finder.get_cluster_leader(vertex_from)
                cluster_to = cluster_finder.get_cluster_leader(vertex_to)
                if cluster_from != cluster_to:
                    cluster_edges.append(
                        ClusterEdge(
                            from_v=vertex_from,
                            to_cluster=cluster_to,
                            weight=weight,
                            to_v=vertex_to,
                            from_cluster=cluster_from
                        )
                    )

            min_cluster_edges = {}

            for cluster_edge in cluster_edges:
                to_cluster = cluster_edge.get_to_cluster()
                if to_cluster not in min_cluster_edges or min_cluster_edges[to_cluster].get_weight() > cluster_edge.get_weight():
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