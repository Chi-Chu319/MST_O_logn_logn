from typing import List

from mpi4py import MPI
import numpy as np
from algo.graph import GraphLocal
from algo.quick_union import QuickUnionUF
from algo.utils.graph_util import GraphUtil
import sys

# -----------!!!!!!!!!!!-----------
# If one process exits because of error, other ones will be all waiting for alltoall messages in next step 
# thus any logs will not be printed
# -----------!!!!!!!!!!!-----------


def mst_distributed(comm: MPI.Intracomm, size: int, rank: int, num_vertex_local: int, graph_local: GraphLocal):
    # [rank * num_vertex_local, (rank + 1) * num_vertex_local) 
    vertex_local_start = rank * num_vertex_local
    num_vertex = num_vertex_local * size

    logs = []

    # phase
    k = 0
    '''
    # which cluster each vertex is in (not local vertices)
    # leader of each vertex

    # Option 1 shared memory clusters array
    # Option 2 each node hold a copy of the array and update it end of each phase (current solution)
    '''

    # contains the vertices in a cluster if a local vertex is a leader of a cluster if not [] (The vertices are sorted by indices)
    clusters_local = [[i + vertex_local_start] for i in range(num_vertex_local)]
    num_cluster = num_vertex

    # list rep of clusters in the form of forest. Same format as Quick union
    cluster_finder_id: List[int] = np.arange(0, num_vertex).tolist()
    mst_edges = []
    cluster_finder = QuickUnionUF(cluster_finder_id)

    while True:
        t_seq_start = MPI.Wtime()

        mpi_time = 0
        t_start_all = MPI.Wtime()

        if rank == 0:
            print(f"cluster_finder size: {sys.getsizeof(cluster_finder.id)}")
            print(f"clusters_local size: {sum([sys.getsizeof(edges) for edges in clusters_local])}")

        cluster_finder.set_id(cluster_finder_id)
        cluster_finder.reset_finished()
        # Step 1
        # O(d logn) d is the degree
        sendbuf_to_clusters = GraphUtil.get_min_weight_to_cluster_edges(graph_local, cluster_finder)
        
        t_seq_end = MPI.Wtime()

        if rank == 0:
            print(f"1 {t_seq_end - t_seq_start}")


        if rank == 0:
            print(f"sendbuf_to_clusters size: {sum([sys.getsizeof(edges) for edges in sendbuf_to_clusters])}")

        '''
        ------------------------------------------------
        '''
        t_start = MPI.Wtime()
        recvbuf_to_clusters = comm.alltoall(sendbuf_to_clusters)
        t_end = MPI.Wtime()
        mpi_time += t_end - t_start
        '''
        ------------------------------------------------
        '''
        t_seq_start = MPI.Wtime()

        if rank == 0:
            print(f"recvbuf_to_clusters size: {sum([sys.getsizeof(edges) for edges in recvbuf_to_clusters])}")

        # Step 2
        clusters_edges = [[] for _ in range(num_vertex_local)]
        for edges in recvbuf_to_clusters:
            for edge in edges:
                clusters_edges[cluster_finder.get_cluster_leader(edge[1]) - vertex_local_start].append(edge)


        if rank == 0:
            print(f"clusters_edges size: {sum([sys.getsizeof(edges) for edges in clusters_edges])}")

        sendbuf_from_clusters = [[] for _ in range(size)]

        for vertex_local, cluster_edges in enumerate(clusters_edges):
            if len(cluster_edges) == 0:
                continue

            # A(F)
            # Worst case could be O(n'^2) edges where n' is the cluster size
            min_weight_cluster_edges = sorted(GraphUtil.get_min_weight_from_cluster_edges(cluster_edges, cluster_finder))

            mu = min(len(clusters_local[vertex_local]), num_cluster, len(min_weight_cluster_edges))

            min_weight_cluster_edges = min_weight_cluster_edges[:mu]

            # Scatter the edges from each cluster leader to cluster members
            for cluster_vertex_idx, cluster_vertex in enumerate(clusters_local[vertex_local]):
                if cluster_vertex_idx < len(min_weight_cluster_edges):
                    edge = min_weight_cluster_edges[cluster_vertex_idx]
                    sendbuf_from_clusters[graph_local.get_vertex_machine(cluster_vertex)].append(edge)

        t_seq_end = MPI.Wtime()

        if rank == 0:
            print(f"2 {t_seq_end - t_seq_start}")

        if rank == 0:
            print(f"sendbuf_from_clusters size: {sum([sys.getsizeof(edges) for edges in sendbuf_from_clusters])}")

        '''
        ------------------------------------------------
        '''
        t_start = MPI.Wtime()
        recvbuf_from_clusters = comm.alltoall(sendbuf_from_clusters)
        t_end = MPI.Wtime()
        mpi_time += t_end - t_start
        '''
        ------------------------------------------------
        '''
        t_seq_start = MPI.Wtime()


        if rank == 0:
            print(f"recvbuf_from_clusters size: {sum([sys.getsizeof(edges) for edges in recvbuf_from_clusters])}")

        # Step 3
        # Each cluster member send edges to v_0
        guardian_cluster_edges = []
        for edges in recvbuf_from_clusters:
            guardian_cluster_edges.extend(edges)

        if rank == 0:
            print(f"guardian_cluster_edges size: {sys.getsizeof(guardian_cluster_edges)}")


        t_seq_end = MPI.Wtime()

        if rank == 0:
            print(f"3 {t_seq_end - t_seq_start}")

        '''
        ------------------------------------------------
        '''
        t_start = MPI.Wtime()
        gathered_edges = comm.gather(guardian_cluster_edges, root=0)
        t_end = MPI.Wtime()
        mpi_time += t_end - t_start
        '''
        ------------------------------------------------
        '''

        t_seq_start = MPI.Wtime()


        if rank == 0:
            print(f"gathered_edges size: {sum([sys.getsizeof(edges) for edges in gathered_edges])}")


        # Step 4
        if rank == 0:
            edges_to_add = []
            for edges in gathered_edges:
                edges_to_add.extend(edges)

            edges_to_add = sorted(edges_to_add)
            added_edges = []
            heaviest_edges = [False] * len(edges_to_add)

            encountered_clusters = {}

            for edge_idx in range(len(edges_to_add) - 1, -1, -1):
                edge = edges_to_add[edge_idx]
                to_cluster = cluster_finder.get_cluster_leader(edge[1])
                if to_cluster in encountered_clusters:
                    heaviest_edges[edge_idx] = True
                    encountered_clusters[to_cluster] = True

            print(f"edges_to_add size: {sys.getsizeof(edges_to_add)}")

            for edge in edges_to_add:
                from_cluster = cluster_finder.get_cluster_leader(edge[0])
                to_cluster = cluster_finder.get_cluster_leader(edge[1])

                # check if dangerous
                if cluster_finder.is_finished(to_cluster):
                    continue

                merged = cluster_finder.safe_union(from_cluster, to_cluster)

                if merged:
                    added_edges.append(edge)
                    if heaviest_edges[edge_idx]:
                        # both finished
                        cluster_finder.set_finished(from_cluster)
                elif (not merged) and heaviest_edges[edge_idx]:
                    cluster_finder.set_finished(to_cluster)

            print(f"added_edges size: {sys.getsizeof(added_edges)}")

            mst_edges.extend(added_edges)
            cluster_finder.flatten()

        cluster_finder_id = cluster_finder.get_id()
        num_cluster = len(set(cluster_finder_id))

        t_seq_end = MPI.Wtime()

        if rank == 0:
            print(f"4 {t_seq_end - t_seq_start}")

        '''
        ------------------------------------------------
        '''
        t_start = MPI.Wtime()
        num_cluster = comm.bcast(num_cluster, root=0)
        cluster_finder_id = comm.bcast(cluster_finder_id, root=0)
        t_end = MPI.Wtime()
        mpi_time += t_end - t_start
        '''
        ------------------------------------------------
        '''

        t_seq_start = MPI.Wtime()

        clusters_local = [[] for _ in range(num_vertex_local)]
        for vertex, cluster_leader in enumerate(cluster_finder_id):
            if vertex_local_start <= cluster_leader < vertex_local_start + num_vertex_local:
                clusters_local[cluster_leader - vertex_local_start].append(vertex)

        # TODO check barrier (is necessary?) after all collective communication
        if k >= 10:
            raise Exception("k reaches 10")

        k += 1

        t_seq_end = MPI.Wtime()

        if rank == 0:
            print(f"5 {t_seq_end - t_seq_start}")

        t_end_all = MPI.Wtime()
        logs.append((t_end_all - t_start_all, mpi_time))

        if num_cluster == 1:
            break

    return mst_edges, k, logs
