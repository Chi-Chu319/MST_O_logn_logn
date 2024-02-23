from mpi4py import MPI
import sys
from algo.graph import Graph
from tester import seq_vs_dist

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

num_vertex_local = int(sys.argv[1])
expected_degree = int(sys.argv[2])
max_weight = int(sys.argv[3])

# Generating graph
if rank == 0:
    # TODO change to distributed graph gen (think about generating a matrix in distributed setting)
    graph = Graph(
        comm_size=size,
        expected_degree=expected_degree,
        max_weights=max_weight,
        num_vertex_local=num_vertex_local
    )
    graph.generate()
else:
    graph = None

seq_vs_dist(graph, comm, rank, size, num_vertex_local)

