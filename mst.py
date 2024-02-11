from mpi4py import MPI
import numpy as np
import sys

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# for graph
num_vertices_per_machine = int(sys.argv[1])
expected_degree = int(sys.argv[2])

num_vertices = size * num_vertices_per_machine
# [rank * num_vertices_per_machine, (rank + 1) * num_vertices_per_machine) 
vertex_index_start = rank * num_vertices_per_machine

edges = []
for vertex_local in range(num_vertices_per_machine):
  vertex_edges = np.random.binomial(1, expected_degree / (num_vertices - 1), num_vertices - 1)
  vertex_edges = np.where(vertex_edges == 1)[0]
  
  vertex_edges += np.where(vertex_edges >= vertex_local + vertex_index_start, 1, 0)


  edges.append(vertex_edges)

# Debugging graph generation 
# for vertex_local in range(num_vertices_per_machine):
#   print(f'vertex_local: {vertex_local + vertex_index_start}, edges: {edges[vertex_local]}')


# Algo