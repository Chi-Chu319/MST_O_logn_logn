import sys

import pandas as pd
from mpi4py import MPI

from algo.graph import Graph, GraphLocal
from algo.mst_distributed import mst_distributed
from algo.mst_sequential import mst_sequential
from algo.utils.graph_util import GraphUtil
from algo.utils.log_util import LogUtil


def seq_vs_dist(comm: MPI.Intracomm, graph: Graph, rank: int, size: int, num_vertex_local: int):
    """
    Distributed MST
    """
    if rank == 0:
        sendbuf = graph.split()
    else:
        sendbuf = None
    # Scatter vertices
    graph_local: GraphLocal = comm.scatter(sendobj=sendbuf, root=0)
    comm.barrier()
    t_start_dist = MPI.Wtime()

    mst_edges_dist, k_dist, logs_dist = mst_distributed(
        comm=comm,
        rank=rank,
        size=size,
        num_vertex_local=num_vertex_local,
        graph_local=graph_local
    )

    t_end_dist = MPI.Wtime()

    """
    Sequential MST
    """
    t_start_seq = MPI.Wtime()

    if rank == 0:
        mst_seq = mst_sequential(graph)
    else:
        mst_seq = None

    t_end_seq = MPI.Wtime()

    return graph, t_start_seq, t_end_seq, t_start_dist, t_end_dist, mst_seq, mst_edges_dist, k_dist, logs_dist


def single_seq_vs_dist(rank: int, size: int, comm: MPI.Intracomm, max_weight: int, num_vertex_local: int):
    graph = GraphUtil.generate_clique_graph(
        rank=rank,
        comm_size=size,
        max_weight=max_weight,
        num_vertex_local=num_vertex_local
    )

    if rank == 0:
        print(f"graph size: {sys.getsizeof(graph.vertices)}")

    test_result = seq_vs_dist(comm, graph, rank, size, num_vertex_local)
    if rank == 0:
        LogUtil.log_seq_vs_dist(*test_result)


def range_seq_vs_dist(comm: MPI.Intracomm, rank: int, size: int, filename: str):
    data = {}

    k = 0
    k_max = 8
    i = 2
    while k <= k_max:
        num_vertex_local = i

        graph = GraphUtil.generate_clique_graph(
            rank=rank,
            comm_size=size,
            max_weight=10,
            num_vertex_local=i
        )

        graph, t_start_seq, t_end_seq, t_start_dist, t_end_dist, mst_seq, mst_edges_dist, k_dist, logs_dist = seq_vs_dist(
            comm, graph, rank, size, num_vertex_local)

        if rank == 0:
            is_same, weight_sum_seq, weight_sum_dist = LogUtil.is_same_weight(
                graph=graph,
                mst_seq=mst_seq,
                mst_edges_dist=mst_edges_dist
            )
            if not is_same:
                print(f"different results! graph size: {graph.num_vertices}, number of machines: {size}")

            t_seq, t_dist, t_dist_seq, t_dist_mpi = LogUtil.seq_dist_time(
                t_start_seq=t_start_seq,
                t_end_seq=t_end_seq,
                t_start_dist=t_start_dist,
                t_end_dist=t_end_dist,
                logs_dist=logs_dist
            )

            data[str(k)] = (t_seq, t_dist, t_dist_seq, t_dist_mpi, k_dist)

        i = i * 2
        k += 1

    if rank == 0:
        df = pd.DataFrame.from_dict(data, orient='index',
                                    columns=['t_seq', 't_dist', 't_dist_seq', 't_dist_mpi', 'k_dist'])
        df.to_csv(filename)
        print(df)


def range_dist(comm: MPI.Intracomm, rank: int, size: int, filename: str):
    data = {}

    k = 0
    k_max = 8
    i = 2
    while k <= k_max:
        num_vertex_local = i

        graph_local = GraphUtil.generate_distribute_clique_graph(
            comm=comm,
            rank=rank,
            comm_size=size,
            max_weight=10,
            num_vertex_local=num_vertex_local
        )

        t_start_dist = MPI.Wtime()

        mst_edges_dist, k_dist, logs_dist = mst_distributed(
            comm=comm,
            rank=rank,
            size=size,
            num_vertex_local=num_vertex_local,
            graph_local=graph_local
        )
        t_end_dist = MPI.Wtime()

        t_dist, t_dist_seq, t_dist_mpi = LogUtil.dist_time(t_start_dist, t_end_dist, logs_dist)

        data[str(k)] = (t_dist, t_dist_seq, t_dist_mpi, k_dist)

        i = i * 2
        k += 1

    if rank == 0:
        df = pd.DataFrame.from_dict(data, orient='index',
                                    columns=['t_dist', 't_dist_seq', 't_dist_mpi', 'k_dist'])
        df.to_csv(filename)
        print(df)