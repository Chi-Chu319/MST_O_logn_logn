from mpi4py import MPI

from algo.graph import Graph, GraphLocal
from algo.mst_distributed import mst_distributed
from algo.mst_sequential import mst_sequential
from algo.utils.logUtils import LogUtils


def seq_vs_dist(graph: Graph, comm: MPI.Intracomm, rank: int, size: int, num_vertex_local: int):
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

    if rank == 0:
        LogUtils.log_seq_vs_dist(
            graph=graph,
            t_start_seq=t_start_seq,
            t_end_seq=t_end_seq,
            t_start_dist=t_start_dist,
            t_end_dist=t_end_dist,
            mst_seq=mst_seq,
            mst_edges_dist=mst_edges_dist,
            k_dist=k_dist,
            logs_dist=logs_dist
        )
