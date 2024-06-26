from algo.mst_sequential import mst_sequential
from mpi4py import MPI
from algo.graph import Graph
from algo.mst_distributed import mst_distributed
from algo.utils.log_util import LogUtil
from algo.utils.graph_util import GraphUtil
from tester import seq_vs_dist, range_dist, range_seq_vs_dist
import sys
import time
import pandas as pd

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

num_vertex_local = int(sys.argv[1])
expected_degree = int(sys.argv[2])
max_weight = int(sys.argv[3])

# # # benchmark
# range_dist(
#     comm=comm,
#     rank=rank,
#     size=size,
#     k_max=10,
#     filename="dist_n2_t16"
# )


# range_seq_vs_dist(
#     comm=comm,
#     rank=rank,
#     size=size,
#     k_max=10,
#     filename="seq_vs_dist_n1_t8"
# )

# graph = Graph(
#   comm_size=size,
#   num_vertex_local=3,
#   expected_degree=100,
#   max_weight=10,
#   is_clique=True
# )

# result = seq_vs_dist(
#   comm=comm,
#   graph=graph,
#   num_vertex_local=3,
#   rank=rank,
#   size=size
# )

# if rank == 0:
#     LogUtil.log_seq_vs_dist(*result)


# Single dist test
graph_local=GraphUtil.generate_distribute_clique_graph(
  comm=comm, 
  rank=rank,
  comm_size=size,
  max_weight=max_weight,
  num_vertex_local=4096
)

mst_edges_dist, k_dist, logs_dist = mst_distributed(
    comm=comm,
    rank=rank,
    size=size,
    graph_local=graph_local
)
if rank == 0:
    print(logs_dist)
    num_proc = 8
    num_vertex_local = 4096
    t_dist_all = sum([logs_dist[i][0] for i in range(len(logs_dist))])
    t_dist_mpi = sum([logs_dist[i][1] for i in range(len(logs_dist))])
    print(f"{t_dist_all}, {t_dist_all - t_dist_mpi}, {t_dist_mpi}")
    # f = open("strong_scaling/strong_scale.csv", "a")
    # f.write(f"{t_dist_all}, {t_dist_all - t_dist_mpi}, {t_dist_mpi}, {num_proc}, {num_vertex_local}\n")
    # f.close()
    f = open(f"strong_scaling/strong_scale_t{num_proc}_{num_vertex_local}_no_collect.txt", "a")
    f.write(str(logs_dist))
    f.close()

# if rank == 0:
#     data = {}

#     k = 0
#     i = 8
#     k_max = 12
#     while k <= k_max:

#         graph = GraphUtil.generate_clique_graph(
#             rank=0,
#             comm_size=1,
#             max_weight=10,
#             num_vertex_local=i
#         )

#         t_start = time.time()
#         mst_sequential(graph)
#         t_end = time.time()
        
#         data[str(k)] = (graph.num_vertices, t_end - t_start)

#         i = i * 2
#         k += 1

#     df = pd.DataFrame.from_dict(data, orient='index', columns=['num_vertices', 't_prim'])
#     df.to_csv("prim.csv")
# print(df)