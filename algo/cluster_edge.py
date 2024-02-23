

class ClusterEdge:
    def __init__(self, from_v: int, to_v: int, from_cluster: int, to_cluster: int, weight: float) -> None:
        self.from_v = from_v
        self.to_v = to_v
        self.from_cluster = from_cluster
        self.to_cluster = to_cluster
        self.weight = weight
        self.heaviest = False
        self.guardian = -1

    def get_from_v(self) -> int:
        return self.from_v

    def get_to_v(self) -> int:
        return self.to_v

    def get_weight(self) -> float:
        return self.weight

    def set_guardian(self, guardian: int) -> None:
        self.guardian = guardian

    def get_guardian(self) -> int:
        return self.guardian

    def set_heaviest(self, heaviest: bool) -> None:
        self.heaviest = heaviest

    def get_heaviest(self) -> bool:
        return self.heaviest

    def get_from_cluster(self) -> int:
        return self.from_cluster

    def get_to_cluster(self) -> int:
        return self.to_cluster

    def __str__(self) -> str:
        return f'from: {self.from_v}, to: {self.to_v}, from_cluster: {self.from_cluster}, to_cluster: {self.to_cluster}, weight: {str(self.weight)}, guardian: {str(self.guardian)}'

    def __lt__(self, other):
        return self.weight < other.weight

    def __eq__(self, other):
        return self.weight == other.weight
