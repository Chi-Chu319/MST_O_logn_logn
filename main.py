from mpi4py import MPI
from algo.utils.graph_util import GraphUtil
from algo.utils.log_util import LogUtil
from tester import seq_vs_dist
import pandas as pd
import sys

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

num_vertex_local = int(sys.argv[1])
expected_degree = int(sys.argv[2])
max_weight = int(sys.argv[3])

# graph = GraphUtil.generate_clique_graph(
#     rank=rank,
#     comm_size=size,
#     max_weight=max_weight,
#     num_vertex_local=num_vertex_local
# )

# if rank == 0:
#     print(f"graph size: {sys.getsizeof(graph.vertices)}")

# test_result = seq_vs_dist(graph, comm, rank, size, num_vertex_local)
# if rank == 0:
#     LogUtil.log_seq_vs_dist(*test_result)


# ----------------------------

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
        max_weight=max_weight,
        num_vertex_local=num_vertex_local
    )

    # graph = GraphUtil.generate_clique_graph(
    #     rank=rank,
    #     comm_size=size,
    #     max_weight=10,
    #     num_vertex_local=i
    # )

    graph, t_start_seq, t_end_seq, t_start_dist, t_end_dist, mst_seq, mst_edges_dist, k_dist, logs_dist = seq_vs_dist(graph, comm, rank, size, num_vertex_local)

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

    i = i*2
    k += 1

if rank == 0:
    df = pd.DataFrame.from_dict(data, orient='index', columns=['t_seq', 't_dist', 't_dist_seq', 't_dist_mpi', 'k_dist'])
    df.to_csv('seq_vs_dist_n8_t1.csv')
    # df.to_csv('seq_vs_dist_n1_t8.csv')
    print(df)
