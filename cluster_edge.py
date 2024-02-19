from weighted_edge import WeightedEdge


class ClusterEdge:
    def __init__(self, from_v: int, from_cluster: int, to_cluster: int, edge: WeightedEdge) -> None:
        self.from_v = from_v
        self.from_cluster = from_cluster
        self.to_cluster = to_cluster
        self.edge = edge

    def __str__(self) -> str:
        return f'from: {self.from_v}, from_cluster: {self.from_cluster}, to_cluster: {self.to_cluster}, edge: {str(self.edge)}'

    def __lt__(self, other):
        return self.edge < other.edge

    def __eq__(self, other):
        return self.edge == other.edge
