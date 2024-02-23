from mpi4py import MPI
import sys
from algos.mst_distributed import mst_distributed
from algos.graph import Graph, GraphLocal

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

"""
Distributed MST
"""
if rank == 0:
    sendbuf = graph.split()
else:
    sendbuf = None
# Scatter vertices and degrees
graph_local: GraphLocal = comm.scatter(sendobj=sendbuf, root=0)
comm.barrier()
t_start = MPI.Wtime()

mst_distributed(
    comm=comm,
    rank=rank,
    size=size,
    num_vertex_local=num_vertex_local,
    graph_local=graph_local
)

t_end = MPI.Wtime()

"""
Sequential MST
"""