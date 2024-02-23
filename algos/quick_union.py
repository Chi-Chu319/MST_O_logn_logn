from typing import List

import deprecation


class QuickUnionUF:
    # For const_frag
    def __init__(self, clusters: List[int]):
        self.id = [i for i in range(len(clusters))]
        self.cluster = clusters.copy()
        self.sz = [1 for i in range(len(clusters))]
        self.finished = [False for i in range(len(clusters))]

    def root(self, i):
        while i != self.id[i]:
            i = self.id[i]
        return i

    def set_finished(self, p):
        i = self.root(p)
        self.finished[i] = True

    def connected(self, p, q):
        return self.root(p) == self.root(q)

    # def get_leaders(self):
    #     # preprocessing, suppose 1,2 merged to a new cluster ([0, 1, 1]) 1 and later 1 merged with 0, it should be ([0, 0, 0]) but the leader update only happens to update the leader's leader which leads to [0, 0, 1] (1, 2 is a cluster with leader 1)
    #     for idx, i, in self.id:
    #         self.leaders[idx] = self.root[]
    #     return self.leaders

    def get_leaders(self):
        # This does not ensure the leader of a cluster has smallest id but this way the work load is distributed among machines other wise smaller rank machine suffer from more work load 
        leaders = self.cluster.copy()

        for i in range(len(self.id)):
            leaders[i] = self.cluster[self.root(i)]
        
        return leaders

    # def __update_leaders(self, p, q):
    #     i = self.root(p)
    #     j = self.root(q)
    #     if self.leaders[i] < self.leaders[j]:
    #         self.leaders[j] = self.leaders[i]
    #     else:
    #         self.leaders[i] = self.leaders[j]

    def safe_union(self, p, q) -> bool:
        i = self.root(p)
        j = self.root(q)
        if i == j or (self.finished[i] and self.finished[j]):
            return False

        # print(f'merging {self.leaders[i]} and {self.leaders[j]}')
        # print(f'leaders before {self.leaders}')
        # # self.__update_leaders(i, j)
        # print(f'leaders after {self.leaders}')
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
