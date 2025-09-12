#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mcmf_greedy_warm.py

Warm-started MCMF with partial-greedy (留白) and fallback negative-cycle cancellation.

主要功能：
- prune per-user top-K (top_k)
- partial-greedy (max_fill_fraction) 产生 warm-start 分配，但不会填满所有容量
- 将 greedy 分配写入残差图（修改 forward cap, increase reverse cap）
- 检测负成本环并有限次尝试消环（cancel_negative_cycles）
- 调用 MinCostMaxFlow.init_potential + solve，并在异常（如 dijkstra 堆爆）时尝试消环并重试一次
- 最终从残差图重建完整分配与总偏好得分并返回

返回格式（dict）：
{
  "total_flow": int,
  "total_pref_score": int,
  "allocations": { user_id: [(supplier_id, amt), ...], ... },
  "timings": {"greedy_time": float, "mcmf_time": float, "total_time": float}
}

如果抛出异常（例如最终仍不能运行），异常会向上抛出，由外层 caller（如 run_all.py）捕获并记录 traceback。
"""

import time
import copy
import math

INF = 10 ** 18


def prune_instance_topk(inst, k):
    """
    返回一个副本实例，每个用户只保留 top-k 供应商（按 score 降序）。
    """
    if k is None:
        return copy.deepcopy(inst)
    new_inst = {
        "suppliers": copy.deepcopy(inst.get("suppliers", [])),
        "users": [],
        "meta": copy.deepcopy(inst.get("meta", {}))
    }
    for user in inst.get("users", []):
        prefs = user.get("supplier_scores", [])
        prefs_sorted = sorted(prefs, key=lambda x: -int(x[1]))
        kept = prefs_sorted[:k]
        new_inst["users"].append({"id": user["id"], "need": int(user["need"]), "supplier_scores": kept})
    new_inst["meta"]["top_k"] = k
    return new_inst


def run_partial_greedy(inst, max_fill_fraction=0.9):
    """
    在实例上运行 partial greedy。
    - max_fill_fraction: 对每个 user 和每个 supplier，单独限制其被 greedy 使用的最大比例。
      例如 0.9 意味着 greedy 最多使用 90% 的用户需求和 90% 的供应库存（对单个 supplier）。
    返回：
      {"allocations": {user_id: [(sup_id, amt), ...]}, "total_assigned": int, "total_pref_score": int}
    """
    suppliers = inst.get("suppliers", [])
    users = inst.get("users", [])
    # init supplier state
    sup_state = {}
    for s in suppliers:
        orig = int(s.get("stock", 0))
        sup_state[s["id"]] = {"orig": orig, "remaining": orig, "max_use": math.floor(max_fill_fraction * orig)}

    allocations = {u["id"]: [] for u in users}
    total_assigned = 0
    total_pref = 0

    for user in users:
        uid = user["id"]
        need = int(user.get("need", 0))
        max_user_fill = math.floor(max_fill_fraction * need)
        remaining_need = max_user_fill
        prefs = sorted(user.get("supplier_scores", []), key=lambda x: -int(x[1]))
        for sid, score in prefs:
            if remaining_need <= 0:
                break
            if sid not in sup_state:
                continue
            sup = sup_state[sid]
            # supplier per-supplier cap: cannot use more than sup["max_use"] overall; but we track remaining
            avail = max(0, min(sup["remaining"], sup["max_use"]))
            if avail <= 0:
                continue
            take = min(avail, remaining_need)
            if take <= 0:
                continue
            sup["remaining"] -= take
            allocations[uid].append((sid, int(take)))
            total_assigned += int(take)
            total_pref += int(take) * int(score)
            remaining_need -= take
    return {"allocations": allocations, "total_assigned": int(total_assigned), "total_pref_score": int(total_pref)}


def run_mcmf_with_warmstart(inst,
                            top_k=None,
                            use_warmstart=True,
                            max_fill_fraction=0.6,
                            cancel_max_iter=200,
                            cancel_time_limit=1.0,
                            verbose=False):
    """
    Main entry.
    参数:
      - inst: instance dict with "suppliers" and "users"
      - top_k: per-user keep top-k suppliers (pruning)
      - use_warmstart: whether to run greedy warm-start (if False, just run cold MCMF)
      - max_fill_fraction: partial-greedy fill fraction (0..1)
      - cancel_max_iter, cancel_time_limit: limits for negative-cycle cancellation
      - verbose: print debug info
    返回:
      {"total_flow": int, "total_pref_score": int, "allocations": {...}, "timings": {...}}
    抛出异常时由外层捕获并记录 traceback。
    """
    from mcmf import MinCostMaxFlow, detect_negative_cycle, cancel_negative_cycles

    inst_proc = prune_instance_topk(inst, top_k) if top_k is not None else copy.deepcopy(inst)
    suppliers = inst_proc.get("suppliers", [])
    users = inst_proc.get("users", [])
    S = len(suppliers)
    U = len(users)
    n = 1 + S + U + 1
    s = 0
    t = n - 1

    # build MCMF residual graph and record indices for edges we will adjust
    mcmf = MinCostMaxFlow(n)
    supplier_id_to_index = {}
    s_to_sup_edge_idx = {}
    for idx, sup in enumerate(suppliers):
        node_id = 1 + idx
        supplier_id_to_index[sup["id"]] = node_id
        idx1 = len(mcmf.graph[s])
        mcmf.add_edge(s, node_id, int(sup.get("stock", 0)), 0)
        s_to_sup_edge_idx[node_id] = idx1

    user_id_to_index = {}
    user_to_t_edge_idx = {}
    for idx, user in enumerate(users):
        node_id = 1 + S + idx
        user_id_to_index[user["id"]] = node_id
        idx1 = len(mcmf.graph[node_id])
        mcmf.add_edge(node_id, t, int(user.get("need", 0)), 0)
        user_to_t_edge_idx[node_id] = idx1

    sup_user_edge_idx = {}
    for user in users:
        u_node = user_id_to_index[user["id"]]
        for sid, score in user.get("supplier_scores", []):
            if sid in supplier_id_to_index:
                s_node = supplier_id_to_index[sid]
                idx1 = len(mcmf.graph[s_node])
                mcmf.add_edge(s_node, u_node, 10 ** 9, -int(score))
                sup_user_edge_idx[(s_node, u_node)] = idx1

    greedy_time = 0.0
    greedy_res = None

    # run partial greedy to get warm allocations (but do not write into final allocations yet)
    if use_warmstart:
        t0 = time.time()
        greedy_res = run_partial_greedy(inst_proc, max_fill_fraction=max_fill_fraction)
        greedy_time = time.time() - t0
        if verbose:
            print(
                f"[warmstart] greedy assigned {greedy_res['total_assigned']} pref {greedy_res['total_pref_score']} in {greedy_time:.4f}s")

        # apply greedy allocation into residual graph (modify forward caps and reverse caps)
        for uid, lst in greedy_res["allocations"].items():
            u_node = user_id_to_index.get(uid)
            if u_node is None:
                continue
            for sid, amt in lst:
                s_node = supplier_id_to_index.get(sid)
                if s_node is None:
                    continue
                use_amt = int(amt)
                # adjust s -> supplier edge
                if s_node in s_to_sup_edge_idx:
                    eidx = s_to_sup_edge_idx[s_node]
                    fwd = mcmf.graph[s][eidx]
                    rev_idx = fwd[3]
                    # clamp
                    use_amt = min(use_amt, fwd[1])
                    if use_amt > 0:
                        fwd[1] -= use_amt
                        mcmf.graph[s_node][rev_idx][1] += use_amt
                else:
                    # fallback scan
                    for ei, e in enumerate(mcmf.graph[s]):
                        if e[0] == s_node:
                            use_amt = min(use_amt, e[1])
                            e[1] -= use_amt
                            mcmf.graph[s_node][e[3]][1] += use_amt
                            break

                # adjust supplier -> user edge
                key = (s_node, u_node)
                if key in sup_user_edge_idx:
                    eidx = sup_user_edge_idx[key]
                    fwd = mcmf.graph[s_node][eidx]
                    rev_idx = fwd[3]
                    use_amt2 = min(use_amt, fwd[1])
                    if use_amt2 > 0:
                        fwd[1] -= use_amt2
                        mcmf.graph[u_node][rev_idx][1] += use_amt2
                else:
                    # the edge may have been pruned; skip
                    continue

                # adjust user -> t edge
                if u_node in user_to_t_edge_idx:
                    eidx = user_to_t_edge_idx[u_node]
                    fwd = mcmf.graph[u_node][eidx]
                    rev_idx = fwd[3]
                    use_amt3 = min(use_amt, fwd[1])
                    if use_amt3 > 0:
                        fwd[1] -= use_amt3
                        mcmf.graph[t][rev_idx][1] += use_amt3
                else:
                    # fallback scan
                    for ei, e in enumerate(mcmf.graph[u_node]):
                        if e[0] == t:
                            use_amt3 = min(use_amt, e[1])
                            e[1] -= use_amt3
                            mcmf.graph[t][e[3]][1] += use_amt3
                            break

    # optional: if greedy filled everything, still attempt negative-cycle cancellation to improve cost
    reduce_neg_t0 = None
    reduce_neg_t = None
    try:
        # If detect negative cycle -> try to cancel some
        reduce_neg_t0 = time.time()
        if detect_negative_cycle(mcmf, src=s):
            if verbose:
                print("[warmstart] negative cycle detected after greedy; attempting cancellation")
            reduced, cnt = cancel_negative_cycles(mcmf, max_iter=cancel_max_iter, time_limit=cancel_time_limit)
            reduce_neg_t = time.time() - reduce_neg_t0
            if verbose:
                print(f"[warmstart] canceled {cnt} cycles, reduced cost by {reduced}")
    except Exception as e:
        # detection shouldn't crash whole flow; log if verbose
        if verbose:
            print("[warmstart] detect/cancel negative cycles raised:", e)

    # Now run MCMF solve; if it fails due to dijkstra heap explosion (RuntimeError), try cancel_negative_cycles and retry once
    total_flow_delta = 0
    total_cost_delta = 0
    mcmf_time = 0.0
    try:
        t_solve0 = time.time()
        # initialize potential and solve (MinCostMaxFlow handles init_potential inside if you call explicitly)
        mcmf.init_potential(s)
        total_flow_delta, total_cost_delta = mcmf.solve(s, t)
        mcmf_time = time.time() - t_solve0
    except RuntimeError as err:
        # likely due to dijkstra pushing too much -> try cancel negative cycles then retry
        if verbose:
            print("[warmstart] mcmf.solve RuntimeError:", err, " -> attempting cycle cancellation and retry")
        try:
            reduce_neg_t0 = time.time()
            reduced, cnt = cancel_negative_cycles(mcmf, max_iter=cancel_max_iter,
                                                  time_limit=cancel_time_limit)
            reduce_neg_t += time.time() - reduce_neg_t0
            if verbose:
                print(f"[warmstart] after cancellation retry: canceled {cnt}, reduced {reduced}")

            # retry solve once
            t_solve0 = time.time()
            mcmf.init_potential(s)
            total_flow_delta, total_cost_delta = mcmf.solve(s, t)
            mcmf_time = time.time() - t_solve0
        except Exception as e2:
            # give up and re-raise the original or new exception to be caught by caller
            if verbose:
                print("[warmstart] retry after cancellation failed:", e2)
            raise

    # Reconstruct final allocations & totals from residual graph (robust)
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
        for (v, cap, cost, rev) in mcmf.graph[sup_node]:
            if 1 + S <= v <= S + U:
                # the reverse edge at v index rev carries the allocated amount
                allocated = mcmf.graph[v][rev][1]
                if allocated > 0:
                    user_id = users[v - 1 - S]["id"]
                    allocations[user_id].append((sup["id"], int(allocated)))
                    total_flow_all += allocated
                    total_pref_all += allocated * score_map.get((sup_node, v), 0)

    total_flow_all = int(total_flow_all)
    total_pref_all = int(total_pref_all)

    result = {
        "total_flow": total_flow_all,
        "total_pref_score": total_pref_all,
        "allocations": allocations,
        "timings": {"greedy_time": greedy_time, "mcmf_time": mcmf_time, "total_time": greedy_time + mcmf_time,
                    "reduce_neg_cycle_time": reduce_neg_t}
    }
    return result


if __name__ == "__main__":
    # Demo CLI for single instance
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--inst", required=True, help="instance json path")
    parser.add_argument("--out", default=None, help="output json path")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--warm", action="store_true", help="disable warm-start")
    parser.add_argument("--max-fill", type=float, default=0.6, help="max_fill_fraction for partial greedy")
    parser.add_argument("--cancel-iter", type=int, default=200)
    parser.add_argument("--cancel-time", type=float, default=100.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.inst, "r", encoding="utf-8") as f:
        inst = json.load(f)

    res = run_mcmf_with_warmstart(inst,
                                  top_k=args.topk,
                                  use_warmstart=args.warm,
                                  max_fill_fraction=args.max_fill,
                                  cancel_max_iter=args.cancel_iter,
                                  cancel_time_limit=args.cancel_time,
                                  verbose=args.verbose)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            json.dump(res, fo, indent=2, ensure_ascii=False)
    else:
        import pprint

        pprint.pprint(res)
