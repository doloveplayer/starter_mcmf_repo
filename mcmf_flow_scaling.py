#!/usr/bin/env python3
"""
mcmf_flow_scaling.py

实现基于容量缩放（capacity / flow scaling）的最小费用最大流求解器（分阶段/分层增广）。
提供函数：run_mcmf_flow_scaling(inst, top_k=None, max_scale=None, verbose=False)

用法示例：
    from mcmf_flow_scaling import run_mcmf_flow_scaling
    inst = json.load(open("instances/test.json"))
    res = run_mcmf_flow_scaling(inst)
    print(res)

策略简述：
1) 构建残差图（与 run_mcmf_on_instance 相同），边的 cost = -score。
2) 计算最大正向容量 max_cap，令 initial Delta = highest power of two <= max_cap（或由参数 max_scale 指定）。
3) 对于 Delta 从大到小：在残差图上只允许使用 cap >= Delta 的边，用 potentials + Dijkstra（约化成本）寻找 s->t 最短增广路径；
   每次沿路径增广 flow = floor(bottleneck / Delta) * Delta（确保以 Delta 的倍数增广），直到找不到可用路径（cap >= Delta）。
4) Delta //= 2，直到 Delta == 0（或 Delta == 1 后阶段结束）。
5) 返回最终从残差图重建的 allocations、total_flow、total_pref_score，以及计时信息。

注意：该实现与原 SSAP 等价（当 Delta 最终降到 1 时），但缩放阶段通常极大减少增广次数。
"""
import heapq
import time
import copy
from math import log2, floor

INF = 10 ** 18

from mcmf import MinCostMaxFlow, detect_negative_cycle, cancel_negative_cycles


def highest_power_of_two_leq(x):
    if x <= 0:
        return 1
    p = 1 << (int(floor(log2(x))))
    return max(1, p)


def dijkstra_with_threshold(graph, potential, n, s, t, threshold):
    """
    Dijkstra on reduced costs, but only traverse edges with cap >= threshold.
    graph: adjacency lists like [[(v, cap, cost, rev_idx), ...], ...]
    potential: list of node potentials (will not be updated here)
    Returns: (reachable, dist, prev_node, prev_edge)
      - reachable: True if t reachable
      - dist: distance array (reduced costs)
      - prev_node & prev_edge: for path reconstruction
    """
    dist = [INF] * n
    prev_node = [-1] * n
    prev_edge = [-1] * n
    dist[s] = 0
    heap = [(0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        for ei, (v, cap, cost, rev) in enumerate(graph[u]):
            if cap < threshold:
                continue
            # reduced cost
            rc = cost + potential[u] - potential[v]
            nd = dist[u] + rc
            if nd < dist[v]:
                dist[v] = nd
                prev_node[v] = u
                prev_edge[v] = ei
                heapq.heappush(heap, (nd, v))
    return (dist[t] < INF, dist, prev_node, prev_edge)


def flow_scaling_solve(mcmf, s, t, initial_delta=None, verbose=False):
    """
    Core flow-scaling solve routine operating on given MinCostMaxFlow instance (residual graph).
    Returns (total_flow_delta, total_cost_delta) where these are total increments made by this routine
    relative to the current residual graph state.
    Note: We will not rely on returned totals for final answer; instead caller should rebuild full allocations from graph.
    """
    n = mcmf.n

    # initialize potentials (Bellman-Ford) to handle negative costs
    mcmf.init_potential(s)

    # find maximal capacity on any forward edge to choose initial delta
    max_cap = 0
    for u in range(n):
        for v, cap, cost, rev in mcmf.graph[u]:
            if cap > max_cap:
                max_cap = cap
    if initial_delta is None:
        delta = highest_power_of_two_leq(max_cap)
    else:
        delta = initial_delta

    total_flow = 0
    total_cost = 0

    if verbose:
        print(f"[scaling] max_cap={max_cap}, initial delta={delta}")

    while delta >= 1:
        if verbose:
            print(f"[scaling] start phase delta={delta}")
        # In each delta-phase, repeatedly find shortest s->t path using only edges with cap >= delta
        while True:
            # Dijkstra on reduced costs with threshold delta
            reachable, dist, prev_node, prev_edge = dijkstra_with_threshold(mcmf.graph, mcmf.potential, n, s, t, delta)
            if not reachable:
                break
            # update potentials: pi[v] += dist[v] for reachable v
            for v in range(n):
                if dist[v] < INF:
                    mcmf.potential[v] += dist[v]
            # find bottleneck (minimum capacity along path)
            # but we will augment in multiples of delta
            cur = t
            bottleneck = INF
            while cur != s:
                u = prev_node[cur]
                ei = prev_edge[cur]
                if u == -1 or ei == -1:
                    bottleneck = 0
                    break
                cap = mcmf.graph[u][ei][1]
                if cap < bottleneck:
                    bottleneck = cap
                cur = u
            if bottleneck <= 0:
                break
            # round down to multiple of delta
            flow = (bottleneck // delta) * delta
            if flow == 0:
                # if the bottleneck is less than delta, no augment of size delta possible
                break
            # apply augmentation along path: update forward and reverse caps and accumulate cost
            path_cost = 0
            cur = t
            while cur != s:
                u = prev_node[cur]
                ei = prev_edge[cur]
                # forward edge
                v, cap, cost, rev = mcmf.graph[u][ei]
                # reduce forward cap
                mcmf.graph[u][ei][1] -= flow
                # increase reverse cap
                mcmf.graph[v][rev][1] += flow
                path_cost += cost
                cur = u
            total_flow += flow
            total_cost += flow * path_cost
            if verbose:
                print(f"[scaling] augmented flow {flow} at cost per unit {path_cost}, total_flow now {total_flow}")
        # reduce scale
        delta //= 2
    return total_flow, total_cost


# ------------------------------------------------------------------
# Wrapper to build graph from instance and call flow-scaling solver
# ------------------------------------------------------------------
def run_mcmf_flow_scaling(inst, top_k=None, verbose=False, initial_delta=None):
    """
    Build residual graph from instance (like run_mcmf_on_instance) and call flow_scaling_solve.
    Returns a dict similar to run_mcmf_on_instance:
      {"total_flow": int, "total_pref_score": int, "allocations": {...}, "timings": {...}}
    Parameters:
      - inst: instance dict (suppliers, users)
      - top_k: if not None, per-user keep only top_k suppliers (pruning)
      - initial_delta: optionally supply initial delta (power of two) to override automatic selection
    """
    inst_proc = copy.deepcopy(inst)
    if top_k is not None:
        # prune per user top-k
        for i, user in enumerate(inst_proc["users"]):
            prefs = user.get("supplier_scores", [])
            prefs_sorted = sorted(prefs, key=lambda x: -int(x[1]))
            inst_proc["users"][i]["supplier_scores"] = prefs_sorted[:top_k]

    suppliers = inst_proc["suppliers"]
    users = inst_proc["users"]
    S = len(suppliers)
    U = len(users)
    n = 1 + S + U + 1
    s = 0
    t = n - 1

    mcmf = MinCostMaxFlow(n)
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

    # add supply->user edges with large capacity and cost = -score
    for user in users:
        u_idx = user_id_to_index[user["id"]]
        for sid, score in user.get("supplier_scores", []):
            if sid in supplier_id_to_index:
                sup_idx = supplier_id_to_index[sid]
                # large capacity (allow splitting)
                mcmf.add_edge(sup_idx, u_idx, 10 ** 9, -int(score))

    try:
        t0 = time.time()
        incr_flow, incr_cost = flow_scaling_solve(mcmf, s, t, initial_delta, verbose=verbose)
        t1 = time.time()
    except RuntimeError as err:
        # likely dijkstra aborted due to too many heap ops => fallback: record error and try a conservative approach
        if verbose:
            print("[warmstart] mcmf.solve aborted:", err)
        # Try fallback: do a short negative-cycle cancellation then try solve again with a smaller max heap ops (or exit)
        try:
            reduced, cnt = cancel_negative_cycles(mcmf, max_iter=500, time_limit=2.0)
            if verbose:
                print(f"[warmstart] after cancellation: reduced {reduced}, cycles {cnt}, retrying solve...")
            # retry solve once
            t0 = time.time()
            incr_flow, incr_cost = flow_scaling_solve(mcmf, s, t, initial_delta, verbose=verbose)
            t1 = time.time() - t0
        except Exception as e2:
            # give up and return error info
            raise RuntimeError(f"mcmf failed after retry: {e2}") from err

    # reconstruct full allocations and totals from residual graph (robust)
    allocations = {u["id"]: [] for u in users}
    total_flow_all = 0.0
    total_pref_all = 0.0
    # build score map
    score_map = {}
    for user in users:
        u_node = user_id_to_index[user["id"]]
        for sid, score in user.get("supplier_scores", []):
            if sid in supplier_id_to_index:
                s_node = supplier_id_to_index[sid]
                score_map[(s_node, u_node)] = int(score)
    for sup in suppliers:
        sup_node = supplier_id_to_index[sup["id"]]
        for v, cap, cost, rev in mcmf.graph[sup_node]:
            if 1 + S <= v <= S + U:
                allocated = mcmf.graph[v][rev][1]
                if allocated > 0:
                    user_id = users[v - 1 - S]["id"]
                    allocations[user_id].append((sup["id"], int(allocated)))
                    total_flow_all += allocated
                    total_pref_all += allocated * score_map.get((sup_node, v), 0)

    total_flow_all = int(total_flow_all)
    total_pref_all = int(total_pref_all)
    timings = {"scaling_time": t1 - t0}
    return {"total_flow": total_flow_all, "total_pref_score": total_pref_all, "allocations": allocations,
            "timings": timings}
