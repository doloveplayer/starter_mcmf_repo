#!/usr/bin/env python3
import heapq

INF = 10 ** 18


def detect_negative_cycle(mcmf, src=None):
    """
    检测残差图中是否存在可达的负成本环（只考虑 cap>0 边）。
    若 src 为 None，则检测任何节点可达的负环（对所有节点初始化 dist=0）。
    返回 True/False。
    """
    n = mcmf.n
    INF = 10 ** 18
    # If src provided, initialize dist[src] = 0 else initialize all zeros (to detect any negative cycle)
    if src is None:
        dist = [0.0] * n
    else:
        dist = [INF] * n
        dist[src] = 0
    parent = [(-1, -1)] * n
    x = -1
    for i in range(n):
        x = -1
        for u in range(n):
            if dist[u] == INF:
                continue
            for ei, (v, cap, cost, rev) in enumerate(mcmf.graph[u]):
                if cap <= 0:
                    continue
                if dist[v] > dist[u] + cost:
                    dist[v] = dist[u] + cost
                    parent[v] = (u, ei)
                    x = v
        if x == -1:
            return False
    # if we reached here x != -1 meaning there is a negative cycle reachable
    return True


def cancel_negative_cycles(mcmf, max_iter=1000, time_limit=None):
    """
    在残差图上尝试消除负成本环（cycle-cancel）。
    返回 (total_reduced_cost_positive, iterations_done)
    total_reduced_cost_positive 表示总共降低的成本（正数表示成本减少）。
    注意：调用应设置合理的 max_iter 与 time_limit 以避免长时间阻塞。
    """
    import time
    n = mcmf.n
    INF = 10 ** 18
    start = time.time()
    iters = 0
    total_reduced = 0.0
    while True:
        if max_iter is not None and iters >= max_iter:
            break
        if time_limit is not None and (time.time() - start) > time_limit:
            break
        # Bellman-Ford detect negative cycle and get a node on it
        dist = [0.0] * n
        parent = [(-1, -1)] * n
        x = -1
        for i in range(n):
            x = -1
            for u in range(n):
                for ei, (v, cap, cost, rev) in enumerate(mcmf.graph[u]):
                    if cap <= 0: continue
                    if dist[v] > dist[u] + cost:
                        dist[v] = dist[u] + cost
                        parent[v] = (u, ei)
                        x = v
            if x == -1:
                break
        if x == -1:
            break
        # walk back n steps to guarantee on cycle
        y = x
        for _ in range(n):
            y, _ = parent[y]
            if y == -1:
                break
        if y == -1:
            break
        # retrieve cycle
        cycle = []
        v = y
        while True:
            u, ei = parent[v]
            cycle.append((u, ei))
            v = u
            if v == y:
                break
        cycle.reverse()
        # compute bottleneck
        delta = INF
        cycle_cost = 0
        for (u, ei) in cycle:
            v, cap, cost, rev = mcmf.graph[u][ei]
            if cap < delta:
                delta = cap
            cycle_cost += cost
        if delta <= 0 or cycle_cost >= 0:
            # not a negative cycle or nothing to do
            break
        # augment along cycle
        for (u, ei) in cycle:
            v, cap, cost, rev = mcmf.graph[u][ei]
            mcmf.graph[u][ei][1] -= delta
            mcmf.graph[v][rev][1] += delta
        total_reduced += -delta * cycle_cost
        iters += 1
    return total_reduced, iters


class MinCostMaxFlow:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.potential = [0] * n

    def add_edge(self, u, v, cap, cost):
        idx1 = len(self.graph[u])
        idx2 = len(self.graph[v])
        # forward edge: [to, cap, cost, rev_idx]
        self.graph[u].append([v, cap, cost, idx2])
        # reverse edge
        self.graph[v].append([u, 0, -cost, idx1])

    def init_potential(self, s):
        dist = [INF] * self.n
        dist[s] = 0
        for _ in range(self.n - 1):
            updated = False
            for u in range(self.n):
                if dist[u] == INF:
                    continue
                for v, cap, cost, rev in self.graph[u]:
                    if cap > 0 and dist[v] > dist[u] + cost:
                        dist[v] = dist[u] + cost
                        updated = True
            if not updated:
                break
        for i in range(self.n):
            if dist[i] < INF:
                self.potential[i] = dist[i]
            else:
                self.potential[i] = 0

    def init_potential_SPFA(self, s):
        import collections
        dist = [INF] * self.n
        dist[s] = 0
        queue = collections.deque([s])
        in_queue = [False] * self.n
        in_queue[s] = True
        count = [0] * self.n
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            for v, cap, cost, rev in self.graph[u]:
                if cap > 0 and dist[v] > dist[u] + cost:
                    dist[v] = dist[u] + cost
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
                        count[v] += 1
                        if count[v] > self.n:
                            raise ValueError("Negative cycle detected")
        for i in range(self.n):
            if dist[i] < INF:
                self.potential[i] = dist[i]
            else:
                self.potential[i] = 0

    def dijkstra(self, s, t, prev_node, prev_edge, dist, max_heap_ops=5_000_00):
        for i in range(self.n):
            dist[i] = INF
            prev_node[i] = -1
            prev_edge[i] = -1
        dist[s] = 0
        heap = [(0, s)]
        heap_ops = 0
        while heap:
            d, u = heapq.heappop(heap)
            heap_ops += 1
            if heap_ops > max_heap_ops:
                raise RuntimeError(
                    f"dijkstra aborted: exceeded max heap operations={max_heap_ops}. Possible negative-cost cycles or bad potentials.")
            if d != dist[u]:
                continue
            for ei, (v, cap, cost, rev) in enumerate(self.graph[u]):
                if cap <= 0:
                    continue
                new_cost = dist[u] + cost + self.potential[u] - self.potential[v]
                if new_cost < dist[v]:
                    dist[v] = new_cost
                    prev_node[v] = u
                    prev_edge[v] = ei
                    heapq.heappush(heap, (dist[v], v))
        for i in range(self.n):
            if dist[i] < INF:
                self.potential[i] += dist[i]
        return dist[t] < INF

    def solve(self, s, t):
        total_flow = 0
        total_cost = 0
        prev_node = [-1] * self.n
        prev_edge = [-1] * self.n
        dist = [INF] * self.n
        while self.dijkstra(s, t, prev_node, prev_edge, dist):
            flow = INF
            v = t
            while v != s:
                u = prev_node[v]
                ei = prev_edge[v]
                if u == -1 or ei == -1:
                    flow = 0
                    break
                cap = self.graph[u][ei][1]
                if cap < flow:
                    flow = cap
                v = u
            if flow == 0 or flow == INF:
                break
            # augment
            v = t
            path_cost = 0
            while v != s:
                u = prev_node[v]
                ei = prev_edge[v]
                edge = self.graph[u][ei]
                edge[1] -= flow
                rev = edge[3]
                self.graph[v][rev][1] += flow
                path_cost += edge[2]
                v = u
            total_flow += flow
            total_cost += flow * path_cost
        return total_flow, total_cost


def run_mcmf_on_instance(inst):
    suppliers = inst.get("suppliers", [])
    users = inst.get("users", [])
    S = len(suppliers)
    U = len(users)
    n = 1 + S + U + 1
    s = 0;
    t = n - 1
    mcmf = MinCostMaxFlow(n)
    supplier_id_to_index = {}
    for i, sup in enumerate(suppliers):
        node = 1 + i
        supplier_id_to_index[sup["id"]] = node
        mcmf.add_edge(s, node, int(sup.get("stock", 0)), 0)

    user_id_to_index = {}
    for j, user in enumerate(users):
        node = 1 + S + j
        user_id_to_index[user["id"]] = node
        mcmf.add_edge(node, t, int(user.get("need", 0)), 0)

    for user in users:
        u_node = user_id_to_index[user["id"]]
        for sid, score in user.get("supplier_scores", []):
            if sid in supplier_id_to_index:
                sup_node = supplier_id_to_index[sid]
                mcmf.add_edge(sup_node, u_node, 10 ** 9, -int(score))

    mcmf.init_potential(s)
    mcmf.solve(s, t)

    allocations = {u["id"]: [] for u in users}
    total_flow_all = 0
    total_pref = 0
    score_map = {}
    for user in users:
        u_node = user_id_to_index[user["id"]]
        for sid, score in user.get("supplier_scores", []):
            if sid in supplier_id_to_index:
                score_map[(supplier_id_to_index[sid], u_node)] = int(score)

    for sup in suppliers:
        s_node = supplier_id_to_index[sup["id"]]
        for v, cap, cost, rev in mcmf.graph[s_node]:
            if 1 + S <= v <= S + U:
                allocated = mcmf.graph[v][rev][1]
                if allocated > 0:
                    user_id = users[v - 1 - S]["id"]
                    allocations[user_id].append((sup["id"], int(allocated)))
                    total_flow_all += allocated
                    total_pref += allocated * score_map.get((s_node, v), 0)

    return {"total_flow": int(total_flow_all), "total_pref_score": int(total_pref), "allocations": allocations}


if __name__ == "__main__":
    # Demo CLI for single instance
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--inst", required=True, help="instance json path")
    parser.add_argument("--out", default=None, help="output json path")
    args = parser.parse_args()

    with open(args.inst, "r", encoding="utf-8") as f:
        inst = json.load(f)

    res = run_mcmf_on_instance(inst)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            json.dump(res, fo, indent=2, ensure_ascii=False)
    else:
        import pprint

        pprint.pprint(res)