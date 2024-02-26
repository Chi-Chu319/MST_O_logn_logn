from mpi4py import MPI
import sys
from algo.utils.graph_util import GraphUtil
from algo.utils.log_util import LogUtil
from tester import seq_vs_dist
import pandas as pd

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

num_vertex_local = int(sys.argv[1])
expected_degree = int(sys.argv[2])
max_weight = int(sys.argv[3])

data = {}

k = 0
k_max = 11
i = 2
while k <= k_max:
    graph = GraphUtil.generate_graph(
        rank=rank,
        comm_size=size,
        expected_degree=(i + 1)*size,
        max_weight=10,
        num_vertex_local=i
    )

    graph, t_start_seq, t_end_seq, t_start_dist, t_end_dist, mst_seq, mst_edges_dist, k_dist, logs_dist = seq_vs_dist(graph, comm, rank, size, num_vertex_local)

    if rank == 0:
        if not LogUtil.is_same_weight(
            graph=graph,
            mst_seq=mst_seq,
            mst_edges_dist=mst_edges_dist
        ):
            print(f"different results! graph size: {graph.num_vertices}, number of machines: {size}")

        t_seq, t_dist, t_dist_seq, t_dist_mpi = seq_dist_time = LogUtil.seq_dist_time(
            t_start_seq=t_start_seq,
            t_end_seq=t_end_seq,
            t_start_dist=t_start_dist,
            t_end_dist=t_end_dist,
            logs_dist=logs_dist
        )

        data[str(k)] = (t_seq, t_dist, t_dist_seq, t_dist_mpi)
    i = i*2
    k += 1

if rank == 0:
    df = pd.DataFrame.from_dict(data, orient='index', columns=['t_seq', 't_dist', 't_dist_seq', 't_dist_mpi'])
    df.to_csv('results.csv')
    print(df)
