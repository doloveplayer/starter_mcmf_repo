#!/usr/bin/env python3
# mcmf_float.py -- float-friendly MinCostMaxFlow and negative-cycle utilities

import heapq
import time

INF = float('inf')
EPS = 1e-9  # floating point tolerance, adjust if your magnitudes are large

def detect_negative_cycle(mcmf, src=None):
    """
    Detect negative-cost cycle reachable in residual graph (consider cap > EPS edges).
    If src is None, initialize all dist=0 to detect any negative cycle in graph.
    Returns True/False.
    """
    n = mcmf.n
    if src is None:
        dist = [0.0] * n
    else:
        dist = [INF] * n
        dist[src] = 0.0
    parent = [(-1, -1)] * n
    x = -1
    for i in range(n):
        x = -1
        for u in range(n):
            if dist[u] == INF:
                continue
            for ei, edge in enumerate(mcmf.graph[u]):
                v, cap, cost, rev = edge
                if cap <= EPS:
                    continue
                # Relax with EPS tolerance
                if dist[v] > dist[u] + cost + EPS:
                    dist[v] = dist[u] + cost
                    parent[v] = (u, ei)
                    x = v
        if x == -1:
            return False
    return True


def cancel_negative_cycles(mcmf, max_iter=1000, time_limit=None):
    """
    Attempt to cancel negative cycles (cycle-cancel). Return (total_reduced_cost_positive, iterations_done).
    Works with float capacities/costs.
    """
    n = mcmf.n
    start = time.time()
    iters = 0
    total_reduced = 0.0
    while True:
        if max_iter is not None and iters >= max_iter:
            break
        if time_limit is not None and (time.time() - start) > time_limit:
            break
        # Bellman-Ford style detection for negative cycle (floating)
        dist = [0.0] * n
        parent = [(-1, -1)] * n
        x = -1
        for i in range(n):
            x = -1
            for u in range(n):
                for ei, edge in enumerate(mcmf.graph[u]):
                    v, cap, cost, rev = edge
                    if cap <= EPS:
                        continue
                    if dist[v] > dist[u] + cost + EPS:
                        dist[v] = dist[u] + cost
                        parent[v] = (u, ei)
                        x = v
            if x == -1:
                break
        if x == -1:
            break
        # ensure we are on the cycle
        y = x
        for _ in range(n):
            y, _ = parent[y]
            if y == -1:
                break
        if y == -1:
            break
        # collect cycle edges
        cycle = []
        v = y
        while True:
            u, ei = parent[v]
            cycle.append((u, ei))
            v = u
            if v == y:
                break
        cycle.reverse()
        # compute bottleneck (min cap) and cycle cost
        delta = INF
        cycle_cost = 0.0
        for (u, ei) in cycle:
            v, cap, cost, rev = mcmf.graph[u][ei]
            if cap < delta:
                delta = cap
            cycle_cost += cost
        # require delta > EPS and cycle_cost < -EPS (strict negative)
        if delta <= EPS or cycle_cost >= -EPS:
            break
        # augment along cycle (reduce forward cap, increase reverse cap)
        for (u, ei) in cycle:
            v, cap, cost, rev = mcmf.graph[u][ei]
            # subtract delta from forward
            mcmf.graph[u][ei][1] = mcmf.graph[u][ei][1] - delta
            # add to reverse
            mcmf.graph[v][rev][1] = mcmf.graph[v][rev][1] + delta
        total_reduced += -delta * cycle_cost
        iters += 1
    return total_reduced, iters


class MinCostMaxFlow:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        # potential for reduced cost; keep floats
        self.potential = [0.0] * n

    def add_edge(self, u, v, cap, cost):
        """
        Add directed edge u->v with capacity cap (float) and cost (float).
        Internally store forward and reverse edges; capacities and costs are floats.
        """
        # ensure floats
        cap = float(cap)
        cost = float(cost)
        idx1 = len(self.graph[u])
        idx2 = len(self.graph[v])
        # forward
        self.graph[u].append([v, cap, cost, idx2])
        # reverse: zero capacity initially, negative cost
        self.graph[v].append([u, 0.0, -cost, idx1])

    def init_potential(self, s):
        """
        Initialize potentials using Bellman-Ford (float-safe).
        Set potential[v] = dist(s, v) if reachable; else keep as 0.0.
        """
        dist = [INF] * self.n
        dist[s] = 0.0
        for _ in range(self.n - 1):
            updated = False
            for u in range(self.n):
                if dist[u] == INF:
                    continue
                for v, cap, cost, rev in self.graph[u]:
                    if cap > EPS and dist[v] > dist[u] + cost + EPS:
                        dist[v] = dist[u] + cost
                        updated = True
            if not updated:
                break
        for i in range(self.n):
            if dist[i] < INF:
                self.potential[i] = float(dist[i])
            else:
                self.potential[i] = 0.0

    def init_potential_SPFA(self, s):
        """
        SPFA variant (may be faster in practice) with floating comparisons.
        Raises ValueError if negative cycle is detected (via counts).
        """
        import collections
        dist = [INF] * self.n
        dist[s] = 0.0
        queue = collections.deque([s])
        in_queue = [False] * self.n
        in_queue[s] = True
        count = [0] * self.n
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            for v, cap, cost, rev in self.graph[u]:
                if cap > EPS and dist[v] > dist[u] + cost + EPS:
                    dist[v] = dist[u] + cost
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
                        count[v] += 1
                        if count[v] > self.n:
                            raise ValueError("Negative cycle detected (SPFA count)")
        for i in range(self.n):
            if dist[i] < INF:
                self.potential[i] = float(dist[i])
            else:
                self.potential[i] = 0.0

    def dijkstra(self, s, t, prev_node, prev_edge, dist, max_heap_ops=5_000_00):
        """
        Dijkstra on reduced costs using potentials; all float-safe.
        Returns True if t reachable (dist[t] < INF), and updates prev_node/prev_edge/dist.
        May raise RuntimeError if max_heap_ops exceeded.
        """
        for i in range(self.n):
            dist[i] = INF
            prev_node[i] = -1
            prev_edge[i] = -1
        dist[s] = 0.0
        heap = [(0.0, s)]
        heap_ops = 0
        while heap:
            d, u = heapq.heappop(heap)
            heap_ops += 1
            if heap_ops > max_heap_ops:
                raise RuntimeError(f"dijkstra aborted: exceeded max heap operations={max_heap_ops}.")
            # skip stale entries with EPS tolerance
            if d > dist[u] + EPS:
                continue
            for ei, (v, cap, cost, rev) in enumerate(self.graph[u]):
                if cap <= EPS:
                    continue
                # reduced cost: cost + potential[u] - potential[v]
                new_cost = dist[u] + cost + self.potential[u] - self.potential[v]
                if new_cost + EPS < dist[v]:
                    dist[v] = new_cost
                    prev_node[v] = u
                    prev_edge[v] = ei
                    heapq.heappush(heap, (dist[v], v))
        # update potentials for visited nodes (add dist)
        for i in range(self.n):
            if dist[i] < INF:
                self.potential[i] = self.potential[i] + dist[i]
        return dist[t] < INF

    def solve(self, s, t):
        """
        Successive shortest augmenting path using Dijkstra on reduced costs.
        Returns total_flow (float), total_cost (float).
        """
        total_flow = 0.0
        total_cost = 0.0
        prev_node = [-1] * self.n
        prev_edge = [-1] * self.n
        dist = [INF] * self.n
        while self.dijkstra(s, t, prev_node, prev_edge, dist):
            # find bottleneck (min residual capacity along path)
            flow = INF
            v = t
            while v != s:
                u = prev_node[v]
                ei = prev_edge[v]
                if u == -1 or ei == -1:
                    flow = 0.0
                    break
                cap = self.graph[u][ei][1]
                if cap < flow:
                    flow = cap
                v = u
            if flow <= EPS or flow == INF:
                break
            # augment
            v = t
            path_cost = 0.0
            while v != s:
                u = prev_node[v]
                ei = prev_edge[v]
                edge = self.graph[u][ei]
                # subtract from forward
                edge[1] = edge[1] - flow
                rev = edge[3]
                # add to reverse
                self.graph[v][rev][1] = self.graph[v][rev][1] + flow
                path_cost += edge[2]
                v = u
            total_flow += flow
            total_cost += flow * path_cost
        return total_flow, total_cost


# helper to run instance (similar to your previous run_mcmf_on_instance but float-aware)
def run_mcmf_on_instance(inst, mapped_cost_mode="negscore"):
    """
    Build MCMF instance from given inst dict and solve.
    mapped_cost_mode: "negscore" (cost = -score) or "mapped100" (cost = 100 - score) etc.
    Returns dict with float totals and allocations (amounts float).
    """
    suppliers = inst.get("suppliers", [])
    users = inst.get("users", [])
    S = len(suppliers)
    U = len(users)
    n = 1 + S + U + 1
    s = 0
    t = n - 1
    mcmf = MinCostMaxFlow(n)
    supplier_id_to_index = {}
    # add s->supplier edges (use float stock)
    for i, sup in enumerate(suppliers):
        node = 1 + i
        supplier_id_to_index[sup["id"]] = node
        stock = float(sup.get("stock", 0.0))
        mcmf.add_edge(s, node, stock, 0.0)
    user_id_to_index = {}
    for j, user in enumerate(users):
        node = 1 + S + j
        user_id_to_index[user["id"]] = node
        need = float(user.get("need", 0.0))
        mcmf.add_edge(node, t, need, 0.0)
    # add supplier->user edges
    LARGE_CAP = 1e18
    for user in users:
        u_node = user_id_to_index[user["id"]]
        for sid, score in user.get("supplier_scores", []):
            if sid in supplier_id_to_index:
                sup_node = supplier_id_to_index[sid]
                if mapped_cost_mode == "negscore":
                    cost = -float(score)
                elif mapped_cost_mode == "mapped100":
                    cost = float(100.0 - float(score))
                else:
                    cost = -float(score)
                mcmf.add_edge(sup_node, u_node, LARGE_CAP, cost)
    # initialize potentials and solve
    try:
        mcmf.init_potential_SPFA(s)
    except Exception:
        mcmf.init_potential(s)
    flow, cost = mcmf.solve(s, t)
    # reconstruct allocations from residual (reverse edge's cap)
    allocations = {u["id"]: [] for u in users}
    total_flow_all = 0.0
    total_pref = 0.0
    # prepare score_map
    score_map = {}
    for user in users:
        u_node = user_id_to_index[user["id"]]
        for sid, score in user.get("supplier_scores", []):
            if sid in supplier_id_to_index:
                score_map[(supplier_id_to_index[sid], u_node)] = float(score)
    for sup in suppliers:
        s_node = supplier_id_to_index[sup["id"]]
        for (v, cap, cost_e, rev) in mcmf.graph[s_node]:
            if 1 + S <= v <= S + U:
                # reverse edge at user node v: reverse index rev stores capacity equal to assigned amount
                allocated = mcmf.graph[v][rev][1]
                if allocated > EPS:
                    user_id = users[v - 1 - S]["id"]
                    allocations[user_id].append((sup["id"], float(allocated)))
                    total_flow_all += float(allocated)
                    total_pref += float(allocated) * score_map.get((s_node, v), 0.0)
    result = {
        "total_flow": float(total_flow_all),
        "total_cost": float(cost),
        "total_pref_score": float(total_pref),
        "allocations": allocations
    }
    return result


# If run as script, allow CLI same as before
if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--inst", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--mapped_cost", choices=["negscore", "mapped100"], default="negscore")
    args = p.parse_args()
    with open(args.inst, "r", encoding="utf-8") as f:
        inst = json.load(f)
    res = run_mcmf_on_instance(inst, mapped_cost_mode=args.mapped_cost)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            json.dump(res, fo, indent=2)
    else:
        import pprint
        pprint.pprint(res)
