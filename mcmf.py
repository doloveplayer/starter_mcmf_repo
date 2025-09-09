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
        self.graph[u].append([v, cap, cost, idx2])
        self.graph[v].append([u, 0, -cost, idx1])

    def init_potential(self, s):
        dist = [INF] * self.n
        dist[s] = 0
        for _ in range(self.n - 1):
            updated = False
            for u in range(self.n):
                if dist[u] == INF:
                    continue
                for v, cap, cost, rev_idx in self.graph[u]:
                    if cap > 0 and dist[v] > dist[u] + cost:
                        dist[v] = dist[u] + cost
                        updated = True
            if not updated:
                break
        # replace INF with large number for unreachable nodes to keep potentials finite
        for i in range(self.n):
            self.potential[i] = dist[i] if dist[i] < INF else 0

    def dijkstra(self, s, t, prev_node, prev_edge, dist):
        dist[:] = [INF] * self.n
        dist[s] = 0
        heap = [(0, s)]
        while heap:
            d, u = heapq.heappop(heap)
            if d != dist[u]:
                continue
            for idx, (v, cap, cost, rev_idx) in enumerate(self.graph[u]):
                if cap <= 0:
                    continue
                new_cost = dist[u] + cost + self.potential[u] - self.potential[v]
                if new_cost < dist[v]:
                    dist[v] = new_cost
                    prev_node[v] = u
                    prev_edge[v] = idx
                    heapq.heappush(heap, (dist[v], v))
        for i in range(self.n):
            if dist[i] < INF:
                self.potential[i] += dist[i]
        return dist[t] < INF

    def solve(self, s, t):
        self.init_potential(s)
        total_flow = 0
        total_cost = 0
        prev_node = [-1] * self.n
        prev_edge = [-1] * self.n
        dist = [INF] * self.n
        while self.dijkstra(s, t, prev_node, prev_edge, dist):
            flow = INF
            cur = t
            while cur != s:
                u = prev_node[cur]
                idx = prev_edge[cur]
                cap = self.graph[u][idx][1]
                flow = min(flow, cap)
                cur = u
            cur = t
            path_cost = 0
            while cur != s:
                u = prev_node[cur]
                idx = prev_edge[cur]
                v = self.graph[u][idx][0]
                cost = self.graph[u][idx][2]
                self.graph[u][idx][1] -= flow
                rev_idx = self.graph[u][idx][3]
                self.graph[v][rev_idx][1] += flow
                path_cost += cost
                cur = u
            total_flow += flow
            total_cost += flow * path_cost
        return total_flow, total_cost

def run_mcmf_on_instance(inst):
    suppliers = inst["suppliers"]
    users = inst["users"]
    S = len(suppliers)
    U = len(users)
    total_nodes = 1 + S + U + 1
    s = 0
    t = total_nodes - 1
    mcmf = MinCostMaxFlow(total_nodes)
    supplier_id_to_index = {}
    for idx, sup in enumerate(suppliers):
        node_id = 1 + idx
        supplier_id_to_index[sup["id"]] = node_id
        mcmf.add_edge(s, node_id, int(sup["stock"]), 0)
    user_id_to_index = {}
    for idx, user in enumerate(users):
        node_id = 1 + S + idx
        user_id_to_index[user["id"]] = node_id
        mcmf.add_edge(node_id, t, int(user["need"]), 0)
    # add supplier->user edges with large capacity and negative score as cost
    for user in users:
        u_idx = user_id_to_index[user["id"]]
        for sid, score in user.get("supplier_scores", []):
            if sid in supplier_id_to_index:
                sup_idx = supplier_id_to_index[sid]
                # allow splitting: capacity large, cost = -score
                mcmf.add_edge(sup_idx, u_idx, 10**9, -int(score))
    total_flow, total_cost = mcmf.solve(s, t)
    # reconstruct allocation: for each sup->user forward edge, reverse edge capacity equals allocated
    allocations = {}
    for user in users:
        allocations[user["id"]] = []
    for sup in suppliers:
        sup_node = supplier_id_to_index[sup["id"]]
        for v, cap, cost, rev in mcmf.graph[sup_node]:
            # forward edge to some user node
            if 1 + S <= v <= S + U:
                user_node = v
                # reverse edge at user_node index rev has capacity equal to allocated flow
                allocated = mcmf.graph[user_node][rev][1]
                if allocated > 0:
                    user_id = users[user_node - 1 - S]["id"]
                    allocations[user_id].append((sup["id"], allocated))
    result = {"total_flow": int(total_flow), "total_pref_score": int(-total_cost), "allocations": allocations}
    return result

if __name__ == "__main__":
    import argparse, json, sys
    p = argparse.ArgumentParser()
    p.add_argument("--inst", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    with open(args.inst, "r", encoding="utf-8") as f:
        inst = json.load(f)
    res = run_mcmf_on_instance(inst)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    else:
        print(json.dumps(res, indent=2))
