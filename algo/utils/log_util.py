import math
from typing import List, Tuple

from algo.graph import Graph


class LogUtil:
    @staticmethod
    def validate_tree(mst_seq: List[int]):
        num_vertices = len(mst_seq)
        for v in range(num_vertices):
            counter= 0
            parent = v
            while not parent == 0:
                parent = mst_seq[parent]
                counter += 1
                if counter >= num_vertices:
                    return False

            return True 

    @staticmethod
    def seq_dist_time(t_start_seq: float,
                      t_end_seq: float,
                      t_start_dist: float,
                      t_end_dist: float, logs_dist: List[Tuple[float, int]]):
        """Seq time, dist time, total seq time, total dist time"""
        t_dist_all = sum([logs_dist[i][0] for i in range(len(logs_dist))])
        t_dist_mpi = sum([logs_dist[i][1] for i in range(len(logs_dist))])

        t_dist = t_end_dist - t_start_dist

        return t_end_seq - t_start_seq, t_end_dist - t_start_dist, t_dist - t_dist_mpi, t_dist_mpi

    @staticmethod
    def dist_time(t_start_dist: float,
                  t_end_dist: float,
                  logs_dist: List[Tuple[float, int]]
                  ):
        """dist time, total seq time, total dist time"""
        t_dist_seq = sum([logs_dist[i][0] for i in range(len(logs_dist))])
        t_dist_mpi = sum([logs_dist[i][1] for i in range(len(logs_dist))])

        t_dist = t_end_dist - t_start_dist

        return t_end_dist - t_start_dist, t_dist - t_dist_mpi, t_dist_mpi

    @staticmethod
    def is_same_weight(graph: Graph, mst_seq: List[int], mst_edges_dist: List[Tuple[int, int, int]], ):
        vertices = graph.vertices

        # sorted to avoid floating point precision issue in addition 
        weight_sum_dist = sum(sorted([edge[2] for edge in mst_edges_dist]))
        weight_sum_seq = sum(sorted([vertices[i][mst_seq[i]] for i in range(1, graph.num_vertices)]))

        return weight_sum_seq == weight_sum_dist and (len(mst_seq) - 1) == len(mst_edges_dist), weight_sum_seq, weight_sum_dist

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
            print(f"Weight sum sequential: {sum([graph.vertices[i][mst_seq[i]] for i in range(1, graph.num_vertices)])}")
            print("mst_edges_dist: ", mst_edges_dist)
            print("mst_seq: ", mst_seq)
            print(graph)

        t_seq, t_dist, t_dist_seq, t_dist_mpi = LogUtil.seq_dist_time(
            t_start_seq=t_start_seq,
            t_end_seq=t_end_seq,
            t_start_dist=t_start_dist,
            t_end_dist=t_end_dist,
            logs_dist=logs_dist
        )

        print("-------------------")
        print(f"Graph size {graph.num_vertices}")
        print("-------------------")
        print(f"Sequential MST time: {t_seq}")
        print("-------------------")
        print(f"Distributed MST time: {t_dist}")
        print(f"number of rounds: {k_dist}")
        for i in range(k_dist):
            print(f"round {i}: seq time: {logs_dist[i][0] - logs_dist[i][1]}, mpi time {logs_dist[i][1]}")
        print("")
        print(f"total seq time: {t_dist_seq}")
        print(f"total mpi time: {t_dist_mpi}")
        print("-------------------")
