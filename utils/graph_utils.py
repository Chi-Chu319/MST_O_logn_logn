from ast import List
from .cluster_edge import ClusterEdge

class GraphUtils(object):
  @staticmethod
  def get_min_weight_to_cluster_edges(vertex, vertex_local, edges_local, weights_local, chosen_edges_local, cluster_leaders) -> List[ClusterEdge]:
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

