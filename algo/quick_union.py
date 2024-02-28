from typing import List


class QuickUnionUF:
    # For const_frag
    def __init__(self, clusters: List[int]):
        self.id = clusters.copy()
        self.sz = [1 for i in range(len(clusters))]
        self.finished = [False for i in range(len(clusters))]

    def set_id(self, new_id: List[int]):
        self.id = new_id

    def reset_finished(self):
        self.finished = [False for i in range(len(self.id))]

    def get_id(self):
        return self.id

    def is_finished(self, p):
        return self.finished[self.root(p)]

    def flatten(self):
        new_id = [self.root(i) for i in range(len(self.id))]
        self.id = new_id

    def get_cluster_leader(self, i: int) -> int:
        return self.root(i)

    # O(log n)
    def root(self, i):
        while i != self.id[i]:
            i = self.id[i]
        return i

    def set_finished(self, p):
        i = self.root(p)
        self.finished[i] = True

    def connected(self, p, q):
        return self.root(p) == self.root(q)

    # O(log n)
    def safe_union(self, p, q) -> bool:
        i = self.root(p)
        j = self.root(q)

        if i == j or (self.finished[i] and self.finished[j]):
            return False

        if self.sz[i] < self.sz[j]:
            self.id[i] = j
            self.sz[j] += self.sz[i]
            self.finished[j] = self.finished[j] or self.finished[i]
            return True
        else:
            self.id[j] = i
            self.sz[i] += self.sz[j]
            self.finished[i] = self.finished[j] or self.finished[i]
            return True

