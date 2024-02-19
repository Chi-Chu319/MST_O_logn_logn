class WeightedEdge:

    def __init__(self, to: int, weight: float):
        self.chosen = False
        self.to = to
        self.weight = weight

    def get_to(self) -> int:
        return self.to

    def get_weight(self) -> float:
        return self.weight

    def set_chosen(self, chosen):
        self.chosen = chosen

    def get_chosen(self) -> bool:
        return self.chosen

    def __str__(self) -> str:
        return f'to: {self.to}, weight: {self.weight}'

    def __lt__(self, other):
        return self.weight < other.weight

    def __eq__(self, other):
        return self.weight == other.weight
