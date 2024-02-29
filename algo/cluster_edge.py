

class ClusterEdge:
    def __init__(self, from_v: int, to_v: int, weight: float) -> None:
        self.from_v = from_v
        self.to_v = to_v
        self.weight = weight

    def get_from_v(self) -> int:
        return self.from_v

    def get_to_v(self) -> int:
        return self.to_v

    def get_weight(self) -> float:
        return self.weight

    def __str__(self) -> str:
        return f'from: {self.from_v}, to: {self.to_v}, weight: {str(self.weight)}'

    def __lt__(self, other):
        return self.weight < other.weight

    def __eq__(self, other):
        return self.weight == other.weight
