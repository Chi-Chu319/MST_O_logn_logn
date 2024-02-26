import math
from typing import List, Tuple

from algo.graph import Graph


class LogUtil:
    @staticmethod
    def seq_dist_time(t_start_seq: float,
                      t_end_seq: float,
                      t_start_dist: float,
                      t_end_dist: float, logs_dist: List[Tuple[float, int]]):
        """Seq time, dist time, total seq time, total dist time"""

        return t_start_seq - t_end_seq, t_start_dist - t_end_dist, sum(logs_dist[:][0]), sum(logs_dist[:][1])

    @staticmethod
    def is_same_weight(graph: Graph, mst_seq: List[int], mst_edges_dist: List[Tuple[int, int, int]], ):
        vertices = graph.vertices

        weight_sum_distributed = sum([edge[2] for edge in mst_edges_dist])
        weight_sum_seq = sum([vertices[i][mst_seq[i]] for i in range(1, graph.num_vertices)])

        return math.isclose(weight_sum_distributed, weight_sum_seq)

    @staticmethod
    def log_seq_vs_dist(
            graph: Graph,
            t_start_seq: float,
            t_end_seq: float,
            t_start_dist: float,
            t_end_dist: float,
            mst_seq: List[int],
            mst_edges_dist: List[Tuple[int, int, int]],
            k_dist: int,
            logs_dist: List[Tuple[float, int]]
    ):
        if not LogUtil.is_same_weight(graph, mst_seq, mst_edges_dist):
            print("different results!")
            print(f"Weight sum distributed: {sum([edge[2] for edge in mst_edges_dist])}")
            print(
                f"Weight sum sequential: {sum([graph.vertices[i][mst_seq[i]] for i in range(1, graph.num_vertices)])}")

        print("-------------------")
        print(f"Graph size {graph.num_vertices}")
        print("-------------------")
        print(f"Sequential MST time: {t_end_seq - t_start_seq}")
        print("-------------------")
        print(f"Distributed MST time: {t_end_dist - t_start_dist}")
        print(f"number of rounds: {k_dist}")
        for i in range(k_dist):
            print(f"round {i}: seq time: {logs_dist[i][0]}, mpi time {logs_dist[i][1]}")
        print("")
        print(f"total seq time: {sum(logs_dist[:][0])}")
        print(f"total mpi time: {sum(logs_dist[:][1])}")
        print("-------------------")