from mpi4py import MPI
from algo.graph import Graph
from algo.utils.log_util import LogUtil
from tester import seq_vs_dist, range_dist, range_seq_vs_dist
import sys

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

num_vertex_local = int(sys.argv[1])
expected_degree = int(sys.argv[2])
max_weight = int(sys.argv[3])

# benchmark
# range_dist(
#     comm=comm,
#     rank=rank,
#     size=size,
#     filename="dist_n1_t8"
# )


range_seq_vs_dist(
    comm=comm,
    rank=rank,
    size=size,
    filename="seq_vs_dist_n1_t8"
)

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
