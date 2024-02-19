from typing import List

from cluster_edge import ClusterEdge


class GraphUtils(object):
    @staticmethod
    def get_min_weight_to_cluster_edges(vertex, vertex_local, edges_local, weights_local, chosen_edges_local,
                                        cluster_leaders) -> List[ClusterEdge]:
        vertex_cluster = cluster_leaders[vertex]

        to_clusters = []

        for edge_idx, edge in enumerate(edges_local[vertex_local]):
            # if not chosen and not from the same cluster
            if not cluster_leaders[edge] == vertex_cluster and chosen_edges_local[vertex_local][edge_idx] == 0:
                to_clusters.append(cluster_leaders[edge])

        to_clusters = list(set(to_clusters))

        to_clusters_edges = [None] * len(to_clusters)

        for to_cluster in to_clusters:
            for edge_idx, edge in enumerate(edges_local[vertex_local]):
                weight = weights_local[edge_idx]

                if cluster_leaders[edge] == to_cluster:
                    to_clusters_edges[to_cluster] = ClusterEdge(
                        from_v=vertex,
                        to_v=edge,
                        from_cluster=vertex_cluster,
                        to_cluster=to_cluster,
                        weight=weight
                    )

        return to_clusters_edges

    @staticmethod
    def get_min_weight_from_cluster_edges(cluster_edges: List[ClusterEdge]) -> List[ClusterEdge]:
        from_clusters = []

        for edge in cluster_edges:
            # if not chosen and not from the same cluster
            from_clusters.append(edge.from_cluster)

        from_clusters = list(set(from_clusters))

        from_clusters_edges = []

        for from_cluster in from_clusters:
            min_weight = -1
            from_cluster_edge = None
            for edge in cluster_edges:
                if edge.from_cluster == from_cluster:
                    if min_weight == -1 or min_weight > edge.weight:
                        min_weight = edge.weight
                        from_cluster_edge = edge

            from_clusters_edges.append(from_cluster_edge)

        return from_clusters_edges

    # @staticmethod
    # def remove_duplicate(l: List[T]):