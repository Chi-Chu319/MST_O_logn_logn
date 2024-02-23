class LogUtils:
    @staticmethod
    def log_seq_vs_dist(
            graph,
            t_start_seq,
            t_end_seq,
            t_start_dist,
            t_end_dist,
            mst_seq,
            mst_edges_dist,
            k_dist,
            logs_dist
    ):
        vertices = graph.vertices

        weight_sum_distributed = sum([edge[2] for edge in mst_edges_dist])
        weight_sum_seq = sum([vertices[i][mst_seq[i]] for i in range(1, graph.num_vertices)])

        if weight_sum_distributed != weight_sum_seq:
            print(f"Weight sum distributed: {weight_sum_distributed}")
            print(f"Weight sum sequential: {weight_sum_seq}")

        print("-------------------")
        print(f"Graph size {graph.num_vertices}")
        print("-------------------")
        print(f"Sequential MST time: {t_end_seq - t_start_seq}")
        print("-------------------")
        print(f"Distributed MST time: {t_end_dist - t_start_dist}")
        print(f"number of rounds: {k_dist}")
        for i in range(k_dist):
            print(f"round {i}: seq time: {logs_dist[i][0]}, mpi time {logs_dist[i][1]}")
        print("\n")
        print(f"total seq time: {sum(logs_dist[:][0])}")
        print(f"total mpi time: {sum(logs_dist[:][1])}")
        print("-------------------")