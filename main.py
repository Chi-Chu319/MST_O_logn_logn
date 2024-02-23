from mpi4py import MPI
import sys
from algos.mst_distributed import mst_distributed
from algos.graph import Graph, GraphLocal
from algos.mst_sequential import mst_sequential

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
t_start_dist = MPI.Wtime()

mst_edges_distributed = mst_distributed(
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
    mst_sequential = mst_sequential(graph)

t_end_seq = MPI.Wtime()

# Compare results
if rank == 0:
    vertices = graph.vertices

    weight_sum_distributed = sum([edge[2] for edge in mst_edges_distributed])
    weight_sum_seq = sum([vertices[i][mst_sequential[i]] for i in range(1, graph.num_vertices)])

    assert mst_edges_distributed == mst_sequential
    print(f"Weight sum distributed: {weight_sum_distributed}")
    print(f"Weight sum sequential: {weight_sum_seq}")
    print(f"Sequential MST time: {t_end_seq - t_start_seq}")
    print(f"Distributed MST time: {t_end_dist - t_start_dist}")
    print("MST edges are equal")
