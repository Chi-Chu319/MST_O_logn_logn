from mpi4py import MPI
from algo.graph import Graph
from algo.mst_distributed import mst_distributed
from algo.utils.log_util import LogUtil
from algo.utils.graph_util import GraphUtil
from tester import seq_vs_dist, range_dist, range_seq_vs_dist
import sys

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

num_vertex_local = int(sys.argv[1])
expected_degree = int(sys.argv[2])
max_weight = int(sys.argv[3])

# # benchmark
# range_dist(
#     comm=comm,
#     rank=rank,
#     size=size,
#     k_max=11,
#     filename="dist_n2_t8"
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
  num_vertex_local=num_vertex_local
)

mst_edges_dist, k_dist, logs_dist = mst_distributed(
    comm=comm,
    rank=rank,
    size=size,
    graph_local=graph_local
)
if rank == 0:
    num_proc = 256
    num_vertex_local = 128
    t_dist_all = sum([logs_dist[i][0] for i in range(len(logs_dist))])
    t_dist_mpi = sum([logs_dist[i][1] for i in range(len(logs_dist))])
    print(f"{t_dist_all}, {t_dist_all - t_dist_mpi}, {t_dist_mpi}")
    f = open("weak_scaling/weak_scale.csv", "a")
    f.write(f"{t_dist_all}, {t_dist_all - t_dist_mpi}, {t_dist_mpi}, {num_proc}, {num_vertex_local}\n")
    f.close()
    f = open(f"weak_scaling/weak_scale_t{num_proc}_{num_vertex_local}.text", "a")
    f.write(str(logs_dist))
    f.close()