#!/usr/bin/env python3
import heapq
INF = 10**18

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

    def dijkstra(self, s, t, prev_node, prev_edge, dist):
        for i in range(self.n):
            dist[i] = INF
            prev_node[i] = -1
            prev_edge[i] = -1
        dist[s] = 0
        heap = [(0, s)]
        while heap:
            d, u = heapq.heappop(heap)
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
    s = 0; t = n - 1
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
                mcmf.add_edge(sup_node, u_node, 10**9, -int(score))

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
