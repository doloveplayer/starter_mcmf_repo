#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mcmf_greedy_warm_float.py

基于原 mcmf_greedy_warm.py 的浮点友好版：支持非整数库存/需求（浮点数）。
需要配合浮点友好版本的 mcmf 模块（MinCostMaxFlow/detect_negative_cycle/cancel_negative_cycles）
以获得更稳健的负环检测与 Dijkstra 行为。
"""
import time
import copy
import math
import json

INF = float('inf')
EPS = 1e-9  # 浮点比较容差
LARGE_CAP = 1e18  # 用于 supplier->user 的“无限”容量（浮点）

def prune_instance_topk(inst, k):
    """
    返回一个副本实例，每个用户只保留 top-k 供应商（按 score 降序）。
    保持 need/stock 为浮点（若原为整数也会被转换为 float）。
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
        # 按 score 降序排序，score 可能为字符串/整数/浮点 -> 转为 float 比较
        prefs_sorted = sorted(prefs, key=lambda x: -float(x[1]))
        kept = prefs_sorted[:k]
        new_inst["users"].append({"id": user["id"], "need": float(user.get("need", 0.0)), "supplier_scores": kept})
    new_inst["meta"]["top_k"] = k
    return new_inst


def run_partial_greedy(inst, max_fill_fraction=0.9):
    """
    部分贪心分配（支持浮点）。
    - max_fill_fraction: 对每个 user 和每个 supplier，单独限制其被 greedy 使用的最大比例（浮点）。
    返回：
      {"allocations": {user_id: [(sup_id, amt), ...]}, "total_assigned": float, "total_pref_score": float}
    """
    suppliers = inst.get("suppliers", [])
    users = inst.get("users", [])
    # 初始化 supplier 状态（浮点）
    sup_state = {}
    for s in suppliers:
        orig = float(s.get("stock", 0.0))
        sup_state[s["id"]] = {
            "orig": orig,
            "remaining": orig,
            # max_use 为浮点，上限 = max_fill_fraction * orig
            "max_use": float(max_fill_fraction) * orig
        }

    allocations = {u["id"]: [] for u in users}
    total_assigned = 0.0
    total_pref = 0.0

    for user in users:
        uid = user["id"]
        need = float(user.get("need", 0.0))
        max_user_fill = float(max_fill_fraction) * need
        remaining_need = max_user_fill
        # 按 score 降序选择偏好（支持 float score）
        prefs = sorted(user.get("supplier_scores", []), key=lambda x: -float(x[1]))
        for sid, score in prefs:
            if remaining_need <= EPS:
                break
            if sid not in sup_state:
                continue
            sup = sup_state[sid]
            # 供应商在 greedy 阶段的可用量（浮点）
            avail = max(0.0, min(sup["remaining"], sup["max_use"]))
            if avail <= EPS:
                continue
            take = min(avail, remaining_need)
            if take <= EPS:
                continue
            sup["remaining"] -= take
            allocations[uid].append((sid, float(take)))
            total_assigned += float(take)
            # total_pref 使用原始 score（浮点）
            total_pref += float(take) * float(score)
            remaining_need -= take
    return {"allocations": allocations, "total_assigned": float(total_assigned), "total_pref_score": float(total_pref)}


def run_mcmf_with_warmstart(inst,
                            top_k=None,
                            use_warmstart=True,
                            max_fill_fraction=0.6,
                            cancel_max_iter=200,
                            cancel_time_limit=1.0,
                            verbose=False,
                            mapped_cost_mode="mapped100"):
    """
    主函数入口（带 warm-start 的 MCMF），浮点友好版本。
    新增参数 mapped_cost_mode:
      - "negscore": 供应->用户 边 cost = -score (可能产生负边)
      - "mapped100": cost = 100 - score (非负，便于 Dijkstra 稳定)
    返回:
      {"total_flow": float, "total_pref_score": float, "allocations": {...}, "timings": {...}}
    """
    # 从 mcmf 模块导入（建议使用浮点友好实现）
    from mcmf_float import MinCostMaxFlow, detect_negative_cycle, cancel_negative_cycles

    # 先做 top-k 剪枝（如指定）
    inst_proc = prune_instance_topk(inst, top_k) if top_k is not None else copy.deepcopy(inst)
    suppliers = inst_proc.get("suppliers", [])
    users = inst_proc.get("users", [])
    S = len(suppliers)
    U = len(users)
    # 节点编号： s(0) -> suppliers(1..S) -> users(S+1..S+U) -> t(n-1)
    n = 1 + S + U + 1
    s = 0
    t = n - 1

    # 构建 MCMF 残差图，并记录一些边的索引以便后续修改
    mcmf = MinCostMaxFlow(n)
    supplier_id_to_index = {}
    s_to_sup_edge_idx = {}
    for idx, sup in enumerate(suppliers):
        node_id = 1 + idx
        supplier_id_to_index[sup["id"]] = node_id
        idx1 = len(mcmf.graph[s])
        # s -> supplier 边，cost 为 0，容量为浮点
        cap_val = float(sup.get("stock", 0.0))
        mcmf.add_edge(s, node_id, cap_val, 0.0)
        s_to_sup_edge_idx[node_id] = idx1

    user_id_to_index = {}
    user_to_t_edge_idx = {}
    for idx, user in enumerate(users):
        node_id = 1 + S + idx
        user_id_to_index[user["id"]] = node_id
        idx1 = len(mcmf.graph[node_id])
        cap_val = float(user.get("need", 0.0))
        # user -> t 边，cost 为 0
        mcmf.add_edge(node_id, t, cap_val, 0.0)
        user_to_t_edge_idx[node_id] = idx1

    sup_user_edge_idx = {}
    # 初始将 supplier->user 的 cost 设置为 -score（浮点）或者 mapped（在后面重建时也会使用 mapped）
    for user in users:
        u_node = user_id_to_index[user["id"]]
        for sid, score in user.get("supplier_scores", []):
            if sid in supplier_id_to_index:
                s_node = supplier_id_to_index[sid]
                idx1 = len(mcmf.graph[s_node])
                # 使用 negscore 作为初始（与原实现保持兼容），但这里为浮点
                mcmf.add_edge(s_node, u_node, float(LARGE_CAP), -float(score))
                sup_user_edge_idx[(s_node, u_node)] = idx1

    greedy_time = 0.0
    greedy_res = None

    # --------------------------
    # 不再直接把 greedy 分配写入残差图（避免引入可达的负反向边）
    # 把 greedy 的分配视为“已固定的流”，通过修改 supplier 的 stock 和 user 的 need，
    # 然后重新从剩余量构建一个新的 MCMF 实例（cold start），以避免 residual 中负环问题。
    # --------------------------
    if use_warmstart:
        t0 = time.time()
        greedy_res = run_partial_greedy(inst_proc, max_fill_fraction=max_fill_fraction)
        greedy_time = time.time() - t0
        if verbose:
            print(f"[warmstart] greedy assigned {greedy_res['total_assigned']:.6f} pref {greedy_res['total_pref_score']:.6f} in {greedy_time:.4f}s")

        # 将 greedy allocations 应用到实例的库存/需求上（不直接修改残差图）
        remaining_stock = {sup["id"]: float(sup.get("stock", 0.0)) for sup in suppliers}
        remaining_need = {user["id"]: float(user.get("need", 0.0)) for user in users}

        # 减去 greedy 分配量（注意可能存在被剪枝的边，此时跳过）
        for uid, lst in greedy_res["allocations"].items():
            for sid, amt in lst:
                if sid in remaining_stock and uid in remaining_need:
                    take = float(amt)
                    # clamp（保险）
                    take = min(take, remaining_stock[sid], remaining_need[uid])
                    remaining_stock[sid] -= take
                    remaining_need[uid] -= take
                # 如果 sid/uid 不在 map（被剪枝），就忽略

        # 2) 重新构建 mcmf：用剩余库存和需求（cold start）
        mcmf = MinCostMaxFlow(n)  # 新实例
        # 供应商节点和 user/t 边重新创建（浮点）
        supplier_id_to_index = {}
        s_to_sup_edge_idx = {}
        for idx, sup in enumerate(suppliers):
            node_id = 1 + idx
            supplier_id_to_index[sup["id"]] = node_id
            idx1 = len(mcmf.graph[s])
            cap = float(remaining_stock.get(sup["id"], 0.0))
            # 添加浮点容量（若为 0 仍添加零容量以保持索引一致）
            mcmf.add_edge(s, node_id, cap, 0.0)
            s_to_sup_edge_idx[node_id] = idx1

        user_id_to_index = {}
        user_to_t_edge_idx = {}
        for idx, user in enumerate(users):
            node_id = 1 + S + idx
            user_id_to_index[user["id"]] = node_id
            idx1 = len(mcmf.graph[node_id])
            cap = float(remaining_need.get(user["id"], 0.0))
            mcmf.add_edge(node_id, t, cap, 0.0)
            user_to_t_edge_idx[node_id] = idx1

        sup_user_edge_idx = {}
        for user in users:
            u_node = user_id_to_index[user["id"]]
            for sid, score in user.get("supplier_scores", []):
                if sid in supplier_id_to_index:
                    s_node = supplier_id_to_index[sid]
                    idx1 = len(mcmf.graph[s_node])
                    # 使用 mapped_cost（工程策略），默认 mapped_cost_mode 用 "mapped100"（外部可指定）
                    if mapped_cost_mode == "mapped100":
                        mapped_cost = float(100.0 - float(score))
                    else:
                        mapped_cost = -float(score)
                    # edge cap as LARGE_CAP (float)
                    mcmf.add_edge(s_node, u_node, float(LARGE_CAP), float(mapped_cost))
                    sup_user_edge_idx[(s_node, u_node)] = idx1

        # 现在 mcmf 是一个干净的残差图（没有把 greedy 流直接写入），接下来照常调用 detect/cancel/solve

    # 可选：若 greedy 后存在负环，可尝试消除以改善成本
    reduce_neg_t0 = None
    reduce_neg_t = None
    try:
        reduce_neg_t0 = time.time()
        if detect_negative_cycle(mcmf, src=s):
            if verbose:
                print("[warmstart] negative cycle detected after greedy; attempting cancellation")
            reduced, cnt = cancel_negative_cycles(mcmf, max_iter=cancel_max_iter, time_limit=cancel_time_limit)
            reduce_neg_t = time.time() - reduce_neg_t0
            if verbose:
                print(f"[warmstart] canceled {cnt} cycles, reduced cost by {reduced}")
    except Exception as e:
        # 负环检测不应当导致整个流程崩溃；若发生异常则在 verbose 模式下输出信息
        if verbose:
            print("[warmstart] detect/cancel negative cycles raised:", e)

    # 运行 MCMF 求解；若出现 Dijkstra 堆爆等 RuntimeError，则尝试消环后重试一次
    total_flow_delta = 0.0
    total_cost_delta = 0.0
    mcmf_time = 0.0
    try:
        t_solve0 = time.time()
        # 初始化 potential（浮点）
        try:
            mcmf.init_potential_SPFA(s)
        except Exception:
            mcmf.init_potential(s)
        total_flow_delta, total_cost_delta = mcmf.solve(s, t)
        mcmf_time = time.time() - t_solve0
    except RuntimeError as err:
        # 可能由于 Dijkstra 堆操作过多导致 -> 尝试消环并重试
        if verbose:
            print("[warmstart] mcmf.solve RuntimeError:", err, " -> attempting cycle cancellation and retry")
        try:
            reduce_neg_t0 = time.time()
            reduced, cnt = cancel_negative_cycles(mcmf, max_iter=cancel_max_iter,
                                                  time_limit=cancel_time_limit)
            if reduce_neg_t is None:
                reduce_neg_t = time.time() - reduce_neg_t0
            else:
                reduce_neg_t += time.time() - reduce_neg_t0
            if verbose:
                print(f"[warmstart] after cancellation retry: canceled {cnt}, reduced {reduced}")

            # 重试一次 solve
            t_solve0 = time.time()
            try:
                mcmf.init_potential_SPFA(s)
            except Exception:
                mcmf.init_potential(s)
            total_flow_delta, total_cost_delta = mcmf.solve(s, t)
            mcmf_time = time.time() - t_solve0
        except Exception as e2:
            if verbose:
                print("[warmstart] retry after cancellation failed:", e2)
            # 无法恢复则抛出异常由上层捕获
            raise

    # -------------------------
    # 合并 greedy 分配 与 MCMF 分配，并计算最终统计（均为浮点）
    # -------------------------
    from collections import defaultdict

    greedy_allocs = greedy_res["allocations"] if (greedy_res is not None and "allocations" in greedy_res) else {}
    greedy_assigned = float(greedy_res["total_assigned"]) if (greedy_res is not None and "total_assigned" in greedy_res) else 0.0
    greedy_pref_score = float(greedy_res["total_pref_score"]) if (greedy_res is not None and "total_pref_score" in greedy_res) else 0.0

    # 构建 score_map（用于计算 mcmf 部分的偏好得分）, 仍使用原始 score
    score_map = {}
    for user in users:
        u_node = user_id_to_index[user["id"]]
        for sid, score in user.get("supplier_scores", []):
            if sid in supplier_id_to_index:
                s_node = supplier_id_to_index[sid]
                score_map[(s_node, u_node)] = float(score)

    # user -> (supplier_id -> amount) 聚合表，先填入 greedy 的结果（float）
    user_alloc_map = {u["id"]: defaultdict(float) for u in users}
    for uid, lst in greedy_allocs.items():
        for sid, amt in lst:
            user_alloc_map.setdefault(uid, defaultdict(float))
            user_alloc_map[uid][sid] += float(amt)

    # 遍历 mcmf.graph 得到 mcmf 分配，并累加到聚合表；同时计算 mcmf 部分的偏好得分
    mcmf_pref_part = 0.0
    mcmf_flow_part = 0.0
    for sup in suppliers:
        sup_node = supplier_id_to_index[sup["id"]]
        for edge in mcmf.graph[sup_node]:
            # 尝试解包 (v, cap, cost, rev)
            try:
                v, cap, cost, rev = edge
            except Exception:
                continue
            # 如果 v 是一个 user 节点范围
            if 1 + S <= v <= S + U:
                # 反向边在用户节点 v 的 rev 索引处，其 cap 字段表示反向边当前容量（已分配量）
                allocated = float(mcmf.graph[v][rev][1])
                if allocated > EPS:
                    user_id = users[v - 1 - S]["id"]
                    user_alloc_map.setdefault(user_id, defaultdict(float))
                    user_alloc_map[user_id][sup["id"]] += allocated
                    mcmf_flow_part += allocated
                    mcmf_pref_part += allocated * score_map.get((sup_node, v), 0.0)

    # 将聚合表转为最终 allocations 格式（user_id -> [(supplier_id, amt), ...]）
    allocations = {}
    total_flow_all = 0.0
    for user in users:
        uid = user["id"]
        allocs = []
        if uid in user_alloc_map:
            for sid, amt in user_alloc_map[uid].items():
                if amt > EPS:
                    allocs.append((sid, float(amt)))
                    total_flow_all += float(amt)
        allocations[uid] = allocs

    # 最终偏好得分：greedy 的得分 + mcmf 计算得到的得分
    total_pref_all = float(greedy_pref_score + mcmf_pref_part)

    if reduce_neg_t is None:
        reduce_neg_t = 0.0

    total_time = (greedy_time or 0.0) + (mcmf_time or 0.0)

    result = {
        "total_flow": float(total_flow_all),
        "total_pref_score": float(total_pref_all),
        "allocations": allocations,
        "timings": {
            "greedy_time": float(greedy_time or 0.0),
            "mcmf_time": float(mcmf_time or 0.0),
            "total_time": float(total_time),
            "reduce_neg_cycle_time": float(reduce_neg_t)
        },
        "breakdown": {
            "greedy_assigned": float(greedy_assigned),
            "greedy_pref_score": float(greedy_pref_score),
            "mcmf_assigned": float(mcmf_flow_part),
            "mcmf_pref_score": float(mcmf_pref_part)
        }
    }
    return result


if __name__ == "__main__":
    # 命令行示例：对单个实例运行（浮点/整数实例均支持）
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inst", required=True, help="instance json path")
    parser.add_argument("--out", default=None, help="output json path")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--warm", action="store_true", help="enable warm-start (use --warm to enable)")
    parser.add_argument("--max-fill", type=float, default=0.5, help="max_fill_fraction for partial greedy")
    parser.add_argument("--cancel-iter", type=int, default=200)
    parser.add_argument("--cancel-time", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mapped-cost", choices=["negscore", "mapped100"], default="negscore",
                        help="how to set supplier->user cost: negscore (cost=-score) or mapped100 (cost=100-score)")
    args = parser.parse_args()

    with open(args.inst, "r", encoding="utf-8") as f:
        inst = json.load(f)

    res = run_mcmf_with_warmstart(inst,
                                  top_k=args.topk,
                                  use_warmstart=args.warm,
                                  max_fill_fraction=args.max_fill,
                                  cancel_max_iter=args.cancel_iter,
                                  cancel_time_limit=args.cancel_time,
                                  verbose=args.verbose,
                                  mapped_cost_mode=args.mapped_cost)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            json.dump(res, fo, indent=2, ensure_ascii=False)
    else:
        import pprint
        pprint.pprint(res)
