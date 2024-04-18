from typing import List

from mpi4py import MPI
import numpy as np
from algo.graph import DistGraphLocal
from algo.quick_union import QuickUnionUF
from algo.utils.graph_util import GraphUtil

# -----------!!!!!!!!!!!-----------
# If one process exits because of error, other ones will be all waiting for alltoall messages in next step 
# thus any logs will not be printed
# -----------!!!!!!!!!!!-----------

# TODO add the node added during the first round into log
# TODO total computation time
# TODO does it makes sense to first scatter and collect just to fit the CONGEST model?
# TODO https://groups.google.com/g/mpi4py/c/Ny-16HE3Aus 
def mst_distributed(comm: MPI.Intracomm, size: int, rank: int, graph_local: DistGraphLocal):
    # [rank * num_vertex_local, (rank + 1) * num_vertex_local)
    num_vertex_local = graph_local.get_num_vertex_local()

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

    # contains the vertices in a cluster if a local vertex is a leader of a cluster if not []
    clusters_local = [[i + vertex_local_start] for i in range(num_vertex_local)]
    num_cluster = num_vertex

    # list rep of clusters in the form of forest. Same format as Quick union
    cluster_finder_id: List[int] = np.arange(0, num_vertex).tolist()
    mst_edges = []
    cluster_finder = QuickUnionUF(cluster_finder_id)

    while True: 
        t_comm0 = 0
        t_comm1 = 0
        t_comm2 = 0
        t_comm3 = 0
        t_comm4 = 0
        t_start_all = MPI.Wtime()

        if k == 0:
            guardian_cluster_edges = []
            for vertex_local, edges in enumerate(graph_local.vertices):
                vertex_to = vertex_local + vertex_local_start
                vertex_from = np.argmin(edges)
                weight = edges[vertex_from]

                guardian_cluster_edges.append((vertex_from, vertex_to, weight))

            # comm0
            '''
            ------------------------------------------------
            '''
            t_start = MPI.Wtime()
            gathered_edges = comm.gather(guardian_cluster_edges, root=0)
            t_end = MPI.Wtime()
            t_comm0 = t_end - t_start
            '''
            ------------------------------------------------
            '''
        else:
            # Step 1
            # O(d logn) d is the degree
            sendbuf_to_clusters = GraphUtil.get_min_weight_to_cluster_edges(graph_local, cluster_finder)

            # comm1
            '''
            ------------------------------------------------
            '''
            t_start = MPI.Wtime() 
            recvbuf_to_clusters = comm.alltoall(sendbuf_to_clusters)
            t_end = MPI.Wtime()
            t_comm1 += t_end - t_start
            '''
            ------------------------------------------------
            '''

            # Step 2
            clusters_edges = [[] for _ in range(num_vertex_local)]
            for edges in recvbuf_to_clusters:
                for edge in edges:
                    clusters_edges[cluster_finder.get_cluster_leader(edge[1]) - vertex_local_start].append(edge)

            sendbuf_from_clusters = [[] for _ in range(size)]

            leader_cluster_edges = []

            for vertex_local, cluster_edges in enumerate(clusters_edges):
                if len(cluster_edges) == 0:
                    continue

                # A(F)
                # Worst case could be O(n'^2) edges where n' is the cluster size
                min_weight_cluster_edges = sorted(
                    GraphUtil.get_min_weight_from_cluster_edges(cluster_edges, cluster_finder),
                    key=lambda x: x[2]
                )

                mu = min(len(clusters_local[vertex_local]), len(min_weight_cluster_edges))

                min_weight_cluster_edges = min_weight_cluster_edges[:mu]

                leader_cluster_edges.extend(min_weight_cluster_edges)

                # Scatter the edges from each cluster leader to cluster members
                # for cluster_vertex_idx, cluster_vertex in enumerate(clusters_local[vertex_local]):
                #     if cluster_vertex_idx < len(min_weight_cluster_edges):
                #         edge = min_weight_cluster_edges[cluster_vertex_idx]
                #         sendbuf_from_clusters[graph_local.get_vertex_machine(cluster_vertex)].append(edge)

            # comm2
            # '''
            # ------------------------------------------------
            # '''
            # t_start = MPI.Wtime()
            # recvbuf_from_clusters = comm.alltoall(sendbuf_from_clusters)
            # t_end = MPI.Wtime()
            # t_comm2 += t_end - t_start
            # '''
            # ------------------------------------------------
            # '''

            # Step 3
            # Each cluster member send edges to v_0
            # guardian_cluster_edges = []
            # for edges in recvbuf_from_clusters:
            #     guardian_cluster_edges.extend(edges)

            # comm3
            '''
            ------------------------------------------------
            '''
            t_start = MPI.Wtime()
            gathered_edges = comm.gather(leader_cluster_edges, root=0)
            t_end = MPI.Wtime()
            t_comm3 += t_end - t_start
            '''
            ------------------------------------------------
            '''
        
        # Step 4
        if rank == 0:
            edges_to_add = []
            for edges in gathered_edges:
                edges_to_add.extend(edges)

            edges_to_add = sorted(
                edges_to_add,
                key=lambda x: x[2]
            )
            heaviest_edges = [False] * len(edges_to_add)

            encountered_clusters = {}

            for edge_idx in range(len(edges_to_add) - 1, -1, -1):
                edge = edges_to_add[edge_idx]
                to_cluster = cluster_finder.get_cluster_leader(edge[1])
                if to_cluster not in encountered_clusters:
                    heaviest_edges[edge_idx] = True
                    encountered_clusters[to_cluster] = True

            for edge_idx, edge in enumerate(edges_to_add):
                from_cluster = cluster_finder.get_cluster_leader(edge[0])
                to_cluster = cluster_finder.get_cluster_leader(edge[1])
                from_cluster_finished = cluster_finder.is_finished(from_cluster)
                to_cluster_finished = cluster_finder.is_finished(to_cluster)

                # check if dangerous
                if to_cluster_finished and from_cluster_finished:
                    continue

                merged = cluster_finder.safe_union(from_cluster, to_cluster)

                if merged:
                    mst_edges.append(edge)
                    if heaviest_edges[edge_idx] or (from_cluster_finished or to_cluster_finished):
                        cluster_finder.set_finished(to_cluster)
                        cluster_finder.set_finished(from_cluster)

                if not merged:
                    if heaviest_edges[edge_idx]:
                        cluster_finder.set_finished(to_cluster)

            cluster_finder.flatten()

        cluster_finder_id = cluster_finder.get_id()
        num_cluster = len(set(cluster_finder_id))

        # comm4
        '''
        ------------------------------------------------
        '''
        # TODO bcast does not comply with the congest clique model as it sends nlogn count. change to scatter and all to all if needed
        t_start = MPI.Wtime()
        num_cluster = comm.bcast(num_cluster, root=0)
        cluster_finder_id = comm.bcast(cluster_finder_id, root=0)
        t_end = MPI.Wtime()
        t_comm4 += t_end - t_start
        '''
        ------------------------------------------------
        '''

        cluster_finder.set_id(cluster_finder_id)
        cluster_finder.reset_finished()

        clusters_local = [[] for _ in range(num_vertex_local)]
        for vertex in range(num_vertex):
            cluster_leader = cluster_finder.get_cluster_leader(vertex)
            if graph_local.get_vertex_machine(cluster_leader) == rank:
                clusters_local[cluster_leader - vertex_local_start].append(vertex)

        if k >= 10:
            raise Exception("k reaches 10")

        k += 1
        
        t_end_all = MPI.Wtime()

        t_mpi = t_comm0 + t_comm1 + t_comm2 + t_comm3 + t_comm4
        logs.append((t_end_all - t_start_all, t_mpi, t_comm0, t_comm1, t_comm2, t_comm3, t_comm4))

        if num_cluster == 1:
            break

    return mst_edges, k, logs
