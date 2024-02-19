from mpi4py import MPI
import numpy as np
import sys
from graph import Graph
# from utils.graph_utils import GraphUtils

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

num_vertex_local = int(sys.argv[1])
expected_degree = int(sys.argv[2])
max_weight = int(sys.argv[3])

# Generating graph
if rank == 0:
    graph = Graph(
        comm_size=size,
        expected_degree=expected_degree,
        max_weights=max_weight,
        num_vertex_local=num_vertex_local
    )
    graph.generate()
    vertices, count, displacement = graph.get_vertices()
else:
    vertices, count, displacement = None, None, None

print("__________")
if rank == 0:
    for i, vertex in enumerate(vertices):
        print(f'Vertex: {i}')
        for edge in vertex:
            print(edge)
print("__________")

vertices_partial = None
# Scatter vertices and degrees
comm.Scatterv(sendbuf=[vertices, tuple(count), tuple(displacement), MPI.LI], recvbuf=vertices_partial, root=0)

print(f'rank: {rank}')
for i, vertex in enumerate(vertices_partial):
    print(f'Vertex: {i}')
    for edge in vertex:
        print(edge)
    


# [rank * num_vertex_local, (rank + 1) * num_vertex_local) 
vertex_local_start = rank * num_vertex_local
num_vertex = num_vertex_local * expected_degree


# # phase
# k = 0
# '''
# # which cluster each vertex is in (not local vertices)
# # leader of each vertex
#
# # Option 1 shared memory clusters array
# # Option 2 each node hold a copy of the array and update it end of each phase (current solution)
# '''
#
# # vertex -> cluster leader (will be updated by rank 0 each iter)
# cluster_leaders = np.arange(0, num_vertex)
# cluster_sizes = np.ones(num_vertex)
# num_cluster = num_vertex
#
# t_start = MPI.Wtime()
#
# while True:
#     # Step 1
#     sendbuf = [[] for _ in range(size)]
#
#     for vertex_local in range(0, num_vertex_local):
#         vertex = vertex_local + vertex_local_start
#         vertex_cluster = cluster_leaders[vertex]
#
#         to_clusters_edges = GraphUtils.get_min_weight_to_cluster_edges(
#             vertex_local=vertex_local,
#             vertex=vertex,
#             chosen_edges_local=chosen_edges_local,
#             cluster_leaders=cluster_leaders,
#             edges_local=edges_local,
#             weights_local=weights_local
#         )
#
#         for edge in to_clusters_edges:
#             cluster_leader_machine = GraphGenerator.get_vertex_machine(cluster_leaders[edge.to_cluster])
#
#             sendbuf[cluster_leader_machine].append(edge)
#
#     recvbuf = comm.alltoall(sendbuf)
#
#     # Step 2
#     clusters_edges = [[] for _ in range(num_vertex_local)]
#     for edges in recvbuf:
#         for edge in edges:
#             clusters_edges[edge.to_cluster - vertex_local_start].append(edge)
#
#     for vertex_local, cluster_edges in enumerate(clusters_edges):
#         if len(cluster_edges) == 0:
#             continue
#
#         min_weight_cluster_edges = GraphUtils.get_min_weight_from_cluster_edges(cluster_edges)
#         min_weight_cluster_edges = sorted(min_weight_cluster_edges, key=lambda x: x.weight)
#
#         mu = min(cluster_sizes[vertex_local + vertex_local_start], num_cluster, len(min_weight_cluster_edges))
#         min_weight_cluster_edges = min_weight_cluster_edges[:mu]
#
#         guardians = [edge.to_v for edge in min_weight_cluster_edges]
#
#         zipped_list = list(sorted(zip(guardians, min_weight_cluster_edges)))
#
#         guardians, min_weight_cluster_edges = zip(*zipped_list)
#
#         guardians, min_weight_cluster_edges = list(guardians), zip(min_weight_cluster_edges)
#
#
#     if num_cluster == 1:
#         break
#
#     k += 1
#
# t_end = MPI.Wtime()
