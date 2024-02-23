from typing import List

from mpi4py import MPI
import numpy as np
from algos.graph import GraphLocal
from algos.quick_union import QuickUnionUF
from algos.utils.graph_utils import GraphUtil

# -----------!!!!!!!!!!!-----------
# If one process exits because of error, other ones will be all waiting for alltoall messages in next step 
# thus any logs will not be printed
# -----------!!!!!!!!!!!-----------

def mst_distributed(comm: MPI. Intracomm, size: int, rank: int, num_vertex_local: int, graph_local: GraphLocal):
    # [rank * num_vertex_local, (rank + 1) * num_vertex_local) 
    vertex_local_start = rank * num_vertex_local
    num_vertex = num_vertex_local * size

    # phase
    k = 0
    '''
    # which cluster each vertex is in (not local vertices)
    # leader of each vertex

    # Option 1 shared memory clusters array
    # Option 2 each node hold a copy of the array and update it end of each phase (current solution)
    '''

    # vertex -> cluster leader (will be updated by rank 0 each iter)
    cluster_leaders: List[int] = np.arange(0, num_vertex).tolist()
    # contains the vertex in clusters if a local vertex is a leader of a cluster if not [] (The vertices are sorted by indices)
    clusters_local = [[i + vertex_local_start] for i in range(num_vertex_local)]
    num_cluster = num_vertex

    while True:
        # Step 1
        sendbuf_to_clusters = GraphUtil.get_min_weight_to_cluster_edges(graph_local, cluster_leaders)

        recvbuf_to_clusters = comm.alltoall(sendbuf_to_clusters)
        comm.barrier()

        # Step 2
        clusters_edges = [[] for _ in range(num_vertex_local)]
        for edges in recvbuf_to_clusters:
            for edge in edges:
                clusters_edges[edge.to_cluster - vertex_local_start].append(edge)

        sendbuf_from_clusters = [[] for _ in range(size)]

        for vertex_local, cluster_edges in enumerate(clusters_edges):
            if len(cluster_edges) == 0:
                continue

            # A(F)
            min_weight_cluster_edges = sorted(GraphUtil.get_min_weight_from_cluster_edges(cluster_edges))

            mu = min(len(clusters_local[vertex_local]), num_cluster, len(min_weight_cluster_edges))

            min_weight_cluster_edges = min_weight_cluster_edges[:mu]

            min_weight_cluster_edges[-1].set_heaviest(True)

            # Scatter the edges from each cluster leader to cluster members
            for cluster_vertex_idx, cluster_vertex in enumerate(clusters_local[vertex_local]):
                if cluster_vertex_idx < len(min_weight_cluster_edges):
                    edge = min_weight_cluster_edges[cluster_vertex_idx]
                    edge.set_guardian(cluster_vertex)
                    sendbuf_from_clusters[graph_local.get_vertex_machine(cluster_vertex)].append(edge)

        recvbuf_from_clusters = comm.alltoall(sendbuf_from_clusters)
        comm.barrier()

        # Step 3
        # Each cluster member send edges to v_0
        guardian_cluster_edges = []
        for edges in recvbuf_from_clusters:
            guardian_cluster_edges.extend(edges)

        gathered_edges = comm.gather(guardian_cluster_edges, root=0)
        comm.barrier()

        # Step 4
        if rank == 0:
            edges_to_add = []
            for edges in gathered_edges:
                edges_to_add.extend(edges)

            edges_to_add = sorted(edges_to_add)

            clusters = []
            # cluster_leader -> idx in clusters
            clusters_idx = [-1 for _ in range(num_vertex)]

            for i in range(num_vertex):
                if cluster_leaders[i] == i:
                    clusters.append(i)

            for idx, cluster in enumerate(clusters):
                clusters_idx[cluster] = idx

            union_find = QuickUnionUF(clusters)

            added_edges = []

            for edge in edges_to_add:
                from_cluster = edge.get_from_cluster()
                to_cluster = edge.get_to_cluster()
                from_cluster_idx = clusters_idx[from_cluster]
                to_cluster_idx = clusters_idx[to_cluster]

                merged = union_find.safe_union(from_cluster_idx, to_cluster_idx)

                if merged:
                    added_edges.append(edge)
                    if edge.get_heaviest():
                        # both finished
                        union_find.set_finished(from_cluster_idx)
                elif (not merged) and edge.get_heaviest():
                    union_find.set_finished(to_cluster_idx)

            new_clusters = union_find.get_leaders()
            new_clusters_map = {}

            for idx, cluster in enumerate(clusters):
                new_clusters_map[cluster] = new_clusters[idx]

            for idx, cluster_leader in enumerate(cluster_leaders):
                cluster_leaders[idx] = new_clusters_map[cluster_leader]

        #     TODO scatter edges to guardians
            sendbuf_chosen_edges = [[] for _ in range(size)]
            for edge in added_edges:
                sendbuf_chosen_edges[graph_local.get_vertex_machine(edge.get_guardian())].append(edge)

            num_cluster = len(set(cluster_leaders))
        else:
            sendbuf_chosen_edges = None

        edges_to_add = comm.scatter(sendobj=sendbuf_chosen_edges, root=0)
        comm.barrier()

        # Step 5
        sendbuf_chosen_edge_endpoints = [[] for _ in range(size)]
        for edge_to_add in edges_to_add:
            sendbuf_chosen_edge_endpoints[graph_local.get_vertex_machine(edge_to_add.get_from_v())].append(edge_to_add)
            sendbuf_chosen_edge_endpoints[graph_local.get_vertex_machine(edge_to_add.get_to_v())].append(edge_to_add)

        recvbuf_chosen_edge_endpoints = comm.alltoall(sendbuf_chosen_edge_endpoints)
        comm.barrier()

        edge_added = []
        for edges in recvbuf_chosen_edge_endpoints:
            edge_added.extend(edges)

        for edge in edge_added:
            graph_local.chose_edge(edge.get_from_v(), edge.get_to_v())

        num_cluster = comm.bcast(num_cluster, root=0)
        comm.barrier()
        cluster_leaders = comm.bcast(cluster_leaders, root=0)
        comm.barrier()

        clusters_local = [[] for _ in range(num_vertex_local)]
        for vertex, cluster_leader in enumerate(cluster_leaders):
            if vertex_local_start <= cluster_leader < vertex_local_start + num_vertex_local:
                clusters_local[cluster_leader - vertex_local_start].append(vertex)

        # TODO check barrier (is necessary?) after all collective communication
        k += 1

        if k >= 10:
            break

        if num_cluster == 1:
            break

