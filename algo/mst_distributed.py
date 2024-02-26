from typing import List

from mpi4py import MPI
import numpy as np
from algo.graph import GraphLocal
from algo.quick_union import QuickUnionUF
from algo.utils.graph_util import GraphUtil

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
        mpi_time = 0
        t_start_all = MPI.Wtime()

        cluster_finder.set_id(cluster_finder_id)
        cluster_finder.reset_finished()
        # Step 1
        sendbuf_to_clusters = GraphUtil.get_min_weight_to_cluster_edges(graph_local, cluster_finder)

        '''
        ------------------------------------------------
        '''
        t_start = MPI.Wtime()
        recvbuf_to_clusters = comm.alltoall(sendbuf_to_clusters)
        comm.barrier()
        t_end = MPI.Wtime()
        mpi_time += t_end - t_start
        '''
        ------------------------------------------------
        '''

        # Step 2
        clusters_edges = [[] for _ in range(num_vertex_local)]
        for edges in recvbuf_to_clusters:
            for edge in edges:
                clusters_edges[edge.get_to_cluster() - vertex_local_start].append(edge)


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

        '''
        ------------------------------------------------
        '''
        t_start = MPI.Wtime()
        recvbuf_from_clusters = comm.alltoall(sendbuf_from_clusters)
        comm.barrier()
        t_end = MPI.Wtime()
        mpi_time += t_end - t_start
        '''
        ------------------------------------------------
        '''

        # Step 3
        # Each cluster member send edges to v_0
        guardian_cluster_edges = []
        for edges in recvbuf_from_clusters:
            guardian_cluster_edges.extend(edges)

        '''
        ------------------------------------------------
        '''
        t_start = MPI.Wtime()
        gathered_edges = comm.gather(guardian_cluster_edges, root=0)
        comm.barrier()
        t_end = MPI.Wtime()
        mpi_time += t_end - t_start
        '''
        ------------------------------------------------
        '''

        # Step 4
        if rank == 0:
            edges_to_add = []
            for edges in gathered_edges:
                edges_to_add.extend(edges)

            edges_to_add = sorted(edges_to_add)
            added_edges = []

            for edge in edges_to_add:
                from_cluster = edge.get_from_cluster()
                to_cluster = edge.get_to_cluster()

                merged = cluster_finder.safe_union(from_cluster, to_cluster)

                if merged:
                    added_edges.append(edge)
                    if edge.get_heaviest():
                        # both finished
                        cluster_finder.set_finished(from_cluster)
                elif (not merged) and edge.get_heaviest():
                    cluster_finder.set_finished(to_cluster)

            for edge in added_edges:
                mst_edges.append((edge.get_from_v(), edge.get_to_v(), edge.get_weight()))

            cluster_finder.flatten()

        cluster_finder_id = cluster_finder.get_id()
        num_cluster = len(set(cluster_finder_id))

        '''
        ------------------------------------------------
        '''
        t_start = MPI.Wtime()
        num_cluster = comm.bcast(num_cluster, root=0)
        comm.barrier()
        cluster_finder_id = comm.bcast(cluster_finder_id, root=0)
        comm.barrier()
        t_end = MPI.Wtime()
        mpi_time += t_end - t_start
        '''
        ------------------------------------------------
        '''

        clusters_local = [[] for _ in range(num_vertex_local)]
        for vertex, cluster_leader in enumerate(cluster_finder_id):
            if vertex_local_start <= cluster_leader < vertex_local_start + num_vertex_local:
                clusters_local[cluster_leader - vertex_local_start].append(vertex)

        # TODO check barrier (is necessary?) after all collective communication
        if k >= 15:
            raise Exception("k reaches 15")

        k += 1

        t_end_all = MPI.Wtime()
        logs.append((t_end_all - t_start_all, mpi_time))

        if num_cluster == 1:
            break

    return mst_edges, k, logs
