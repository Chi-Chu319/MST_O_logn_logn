class WeightedEdge:

    def __init__(self, to, weight):
        self.chosen = False
        self.to = to
        self.weight = weight

    def get_to(self):
        return self.to

    def get_weight(self):
        return self.weight

    def set_chosen(self, chosen):
        self.chosen = chosen

    def get_chosen(self):
        return self.chosen

    def __str__(self):
        return f'to: {self.to}, weight: {self.weight}'
