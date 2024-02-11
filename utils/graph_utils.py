import numpy as np

def partition_vertices(num_machines, num_vertices):
  partitioned_vertices = np.random.choice(num_machines, size=num_vertices, replace=True)
  
  # [machine_rank, vertex_array]
  return [
    np.where(partitioned_vertices == i)[0] for i in range(num_machines)
  ]