class ClusterEdge(object):
  def __init__(self, from_v, to_v, from_cluster, to_cluster, weight) -> None:
    self.from_v = from_v
    self.to_v = to_v
    self.from_cluster = from_cluster
    self.to_cluster = to_cluster
    self.weight = weight