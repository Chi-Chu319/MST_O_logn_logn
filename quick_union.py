from typing import List

import deprecation


class QuickUnionUF:
    # For const_frag
    def __init__(self, clusters: List[int]):
        self.id = [i for i in range(len(clusters))]
        self.cluster = clusters.copy()
        self.sz = [1 for i in range(len(clusters))]
        self.finished = [False for i in range(len(clusters))]
        self.leaders = clusters.copy()

    def root(self, i):
        while i != self.id[i]:
            i = self.id[i]
        return i

    def set_finished(self, p):
        i = self.root(p)
        self.finished[i] = True

    def connected(self, p, q):
        return self.root(p) == self.root(q)

    def get_leaders(self):
        return self.leaders

    def __update_leaders(self, p, q):
        i = self.root(p)
        j = self.root(q)
        if self.leaders[i] < self.leaders[j]:
            self.leaders[j] = self.leaders[i]
        else:
            self.leaders[i] = self.leaders[j]

    def safe_union(self, p, q) -> bool:
        i = self.root(p)
        j = self.root(q)
        if i == j or (self.finished[i] and self.finished[j]):
            return False

        self.__update_leaders(i, j)
        if self.sz[i] < self.sz[j]:
            self.id[i] = j
            self.sz[j] += self.sz[i]
            self.finished[j] = self.finished[j] and self.finished[i]
            return True
        else:
            self.id[j] = i
            self.sz[i] += self.sz[j]
            self.finished[i] = self.finished[j] and self.finished[i]
            return True


    @deprecation.deprecated
    def union(self, p, q):
        i = self.root(p)
        j = self.root(q)
        if i == j:
            return
        if self.sz[i] < self.sz[j]:
            self.id[i] = j
            self.sz[j] += self.sz[i]
        else:
            self.id[j] = i
            self.sz[i] += self.sz[j]
