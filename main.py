from mpi4py import MPI
from tester import seq_vs_dist, range_dist
import sys

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

num_vertex_local = int(sys.argv[1])
expected_degree = int(sys.argv[2])
max_weight = int(sys.argv[3])

# benchmark
range_dist(
    comm=comm,
    rank=rank,
    size=size,
    filename="dist_n1_t8"
)
