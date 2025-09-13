#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mcmf_greedy_warm.py

带部分贪心热启动的最小费用最大流（MCMF）实现。

主要功能：
- 对每个用户保留其 top-K 供应商（按得分排序），用于剪枝（prune）
- 使用部分贪心（partial greedy）生成 warm-start 分配，但不把资源全部耗尽（留白）
- 将 greedy 分配写入残差图（修改前向边容量并增加反向边容量）
- 检测负成本环并有限次尝试消环（cancel_negative_cycles）
- 初始化 potential 并调用 MinCostMaxFlow.solve()；若遇到 Dijkstra 堆爆或异常，尝试消环并重试一次
- 最终从残差图重建完整分配与总偏好得分并返回

提示：
- 为了降低负边带来的问题，这里将 supplier->user 边的 cost 设置为 100 - score（score 最大不会超过 100）。
  这样正向边成本接近 0（score 越高成本越低），减少初始负成本边的数量，有利于 scaling 和 Dijkstra 的稳定性。
"""

import time
import copy
import math
import json

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
        # 按 score 降序排序，score 原本可能为字符串，转为 int 比较
        prefs_sorted = sorted(prefs, key=lambda x: -int(x[1]))
        kept = prefs_sorted[:k]
        new_inst["users"].append({"id": user["id"], "need": int(user["need"]), "supplier_scores": kept})
    new_inst["meta"]["top_k"] = k
    return new_inst


def run_partial_greedy(inst, max_fill_fraction=0.9):
    """
    在实例上运行部分贪心分配（partial greedy）。
    - max_fill_fraction: 对每个 user 和每个 supplier，单独限制其被 greedy 使用的最大比例。
      例如 0.9 意味着 greedy 最多使用 90% 的用户需求和 90% 的供应库存（对单个 supplier）。
    返回：
      {"allocations": {user_id: [(sup_id, amt), ...]}, "total_assigned": int, "total_pref_score": int}
    """
    suppliers = inst.get("suppliers", [])
    users = inst.get("users", [])
    # 初始化 supplier 状态
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
        # 按 score 降序选择偏好
        prefs = sorted(user.get("supplier_scores", []), key=lambda x: -int(x[1]))
        for sid, score in prefs:
            if remaining_need <= 0:
                break
            if sid not in sup_state:
                continue
            sup = sup_state[sid]
            # 供应商在 greedy 阶段的可用量，不能超过 sup["remaining"] 和 sup["max_use"]
            avail = max(0, min(sup["remaining"], sup["max_use"]))
            if avail <= 0:
                continue
            take = min(avail, remaining_need)
            if take <= 0:
                continue
            sup["remaining"] -= take
            allocations[uid].append((sid, int(take)))
            total_assigned += int(take)
            # total_pref 使用原始 score（未变化）
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
    主函数入口（带 warm-start 的 MCMF）。
    参数:
      - inst: instance dict，包含 "suppliers" 和 "users"
      - top_k: 每个用户保留 top-k 供应商（用于剪枝）
      - use_warmstart: 是否使用 warm-start（如果 False 则直接冷启动 MCMF）
      - max_fill_fraction: partial greedy 的填充比例（0..1）
      - cancel_max_iter, cancel_time_limit: 负环消除的限制参数
      - verbose: 是否打印调试信息
    返回:
      {"total_flow": int, "total_pref_score": int, "allocations": {...}, "timings": {...}}
    抛出异常时由外层 caller 捕获并记录 traceback。
    """
    # 从 mcmf 模块导入 MinCostMaxFlow 及负环检测/消除函数
    from mcmf import MinCostMaxFlow, detect_negative_cycle, cancel_negative_cycles

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
        # s -> supplier 边，cost 为 0
        mcmf.add_edge(s, node_id, int(sup.get("stock", 0)), 0)
        s_to_sup_edge_idx[node_id] = idx1

    user_id_to_index = {}
    user_to_t_edge_idx = {}
    for idx, user in enumerate(users):
        node_id = 1 + S + idx
        user_id_to_index[user["id"]] = node_id
        idx1 = len(mcmf.graph[node_id])
        # user -> t 边，cost 为 0
        mcmf.add_edge(node_id, t, int(user.get("need", 0)), 0)
        user_to_t_edge_idx[node_id] = idx1

    sup_user_edge_idx = {}
    # 将 supplier->user 的 cost 从 -score 改为 100 - score
    # 这样正向边成本为非负（score 越高成本越低），有利于减少大量负边导致的问题。
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

    # --------------------------
    # 替换：不再直接把 greedy 分配写入残差图（避免引入可达的负反向边）
    # 我们把 greedy 的分配视为“已固定的流”，通过修改 supplier 的 stock 和 user 的 need，
    # 然后重新从剩余量构建一个新的 MCMF 实例（cold start），以避免 residual 中负环问题。
    # --------------------------
    if use_warmstart:
        t0 = time.time()
        greedy_res = run_partial_greedy(inst_proc, max_fill_fraction=max_fill_fraction)
        greedy_time = time.time() - t0
        if verbose:
            print(f"[warmstart] greedy assigned {greedy_res['total_assigned']} pref {greedy_res['total_pref_score']} in {greedy_time:.4f}s")

        # 将 greedy allocations 应用到实例的库存/需求上（不直接修改残差图）
        # 1) 构建 supplier_id -> remaining_stock（初值为原 stock）
        remaining_stock = {sup["id"]: int(sup.get("stock", 0)) for sup in suppliers}
        remaining_need = {user["id"]: int(user.get("need", 0)) for user in users}

        # 减去 greedy 分配量（注意可能存在被剪枝的边，此时跳过）
        for uid, lst in greedy_res["allocations"].items():
            for sid, amt in lst:
                if sid in remaining_stock and uid in remaining_need:
                    take = int(amt)
                    # clamp（保险）
                    take = min(take, remaining_stock[sid], remaining_need[uid])
                    remaining_stock[sid] -= take
                    remaining_need[uid] -= take
                # 如果 sid/uid 不在 map（被剪枝），就忽略

        # 2) 重新构建 mcmf：用剩余库存和需求（cold start）
        mcmf = MinCostMaxFlow(n)  # 新实例
        # 供应商节点和 user/t 边重新创建
        supplier_id_to_index = {}
        s_to_sup_edge_idx = {}
        for idx, sup in enumerate(suppliers):
            node_id = 1 + idx
            supplier_id_to_index[sup["id"]] = node_id
            idx1 = len(mcmf.graph[s])
            cap = remaining_stock.get(sup["id"], 0)
            if cap > 0:
                mcmf.add_edge(s, node_id, int(cap), 0)
            else:
                # 仍然添加零容量边以保持索引结构一致（如果你的实现依赖）
                mcmf.add_edge(s, node_id, 0, 0)
            s_to_sup_edge_idx[node_id] = idx1

        user_id_to_index = {}
        user_to_t_edge_idx = {}
        for idx, user in enumerate(users):
            node_id = 1 + S + idx
            user_id_to_index[user["id"]] = node_id
            idx1 = len(mcmf.graph[node_id])
            cap = remaining_need.get(user["id"], 0)
            if cap > 0:
                mcmf.add_edge(node_id, t, int(cap), 0)
            else:
                mcmf.add_edge(node_id, t, 0, 0)
            user_to_t_edge_idx[node_id] = idx1

        sup_user_edge_idx = {}
        for user in users:
            u_node = user_id_to_index[user["id"]]
            for sid, score in user.get("supplier_scores", []):
                if sid in supplier_id_to_index:
                    s_node = supplier_id_to_index[sid]
                    idx1 = len(mcmf.graph[s_node])
                    # 使用相同的 cost 映射（例如 100 - score）
                    mapped_cost = 100 - int(score)
                    # 注意：若某一边的供应商已无剩余 stock 或用户已无需求，edge cap 会在搜索过程中自然不能被增广
                    mcmf.add_edge(s_node, u_node, 10 ** 9, int(mapped_cost))
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
    total_flow_delta = 0
    total_cost_delta = 0
    mcmf_time = 0.0
    try:
        t_solve0 = time.time()
        # 初始化 potential
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
            # 如果之前 reduce_neg_t 是 None，则赋初值
            if reduce_neg_t is None:
                reduce_neg_t = time.time() - reduce_neg_t0
            else:
                reduce_neg_t += time.time() - reduce_neg_t0
            if verbose:
                print(f"[warmstart] after cancellation retry: canceled {cnt}, reduced {reduced}")

            # 重试一次 solve
            t_solve0 = time.time()
            mcmf.init_potential(s)
            total_flow_delta, total_cost_delta = mcmf.solve(s, t)
            mcmf_time = time.time() - t_solve0
        except Exception as e2:
            if verbose:
                print("[warmstart] retry after cancellation failed:", e2)
            # 无法恢复则抛出异常由上层捕获
            raise

    # -------------------------
    # 合并 greedy 分配 与 MCMF 分配，并计算最终统计
    # -------------------------
    from collections import defaultdict

    # 如果没有执行 greedy（use_warmstart=False），保证相关变量安全
    greedy_allocs = greedy_res["allocations"] if (greedy_res is not None and "allocations" in greedy_res) else {}
    greedy_assigned = int(greedy_res["total_assigned"]) if (greedy_res is not None and "total_assigned" in greedy_res) else 0
    greedy_pref_score = int(greedy_res["total_pref_score"]) if (greedy_res is not None and "total_pref_score" in greedy_res) else 0

    # 构建 score_map（用于计算 mcmf 部分的偏好得分）, 仍使用原始 score
    score_map = {}
    for user in users:
        u_node = user_id_to_index[user["id"]]
        for sid, score in user.get("supplier_scores", []):
            if sid in supplier_id_to_index:
                s_node = supplier_id_to_index[sid]
                score_map[(s_node, u_node)] = int(score)

    # user -> (supplier_id -> amount) 聚合表，先填入 greedy 的结果
    user_alloc_map = {u["id"]: defaultdict(int) for u in users}
    for uid, lst in greedy_allocs.items():
        for sid, amt in lst:
            user_alloc_map.setdefault(uid, defaultdict(int))
            user_alloc_map[uid][sid] += int(amt)

    # 遍历 mcmf.graph 得到 mcmf 分配，并累加到聚合表；同时计算 mcmf 部分的偏好得分
    mcmf_pref_part = 0
    mcmf_flow_part = 0
    for sup in suppliers:
        sup_node = supplier_id_to_index[sup["id"]]
        # 边的元组通常为 (to, cap, cost, rev)
        for edge in mcmf.graph[sup_node]:
            # 兼容不同表示，尽量解包
            try:
                v, cap, cost, rev = edge
            except Exception:
                # 如果不是四元组，跳过（或根据你的实现调整）
                continue
            # 如果 v 是一个 user 节点范围
            if 1 + S <= v <= S + U:
                # 反向边在用户节点 v 的 rev 索引处，其 cap 字段表示反向边当前容量
                # 分配量等于反向边的容量（因为我们从残差图中恢复）
                allocated = mcmf.graph[v][rev][1]
                if allocated > 0:
                    user_id = users[v - 1 - S]["id"]
                    # 将 supplier 的原始 id 加入聚合表
                    user_alloc_map.setdefault(user_id, defaultdict(int))
                    user_alloc_map[user_id][sup["id"]] += int(allocated)
                    mcmf_flow_part += int(allocated)
                    # 用 score_map 计算对应的偏好得分（若缺失则默认 0）
                    mcmf_pref_part += int(allocated) * score_map.get((sup_node, v), 0)

    # 将聚合表转为最终 allocations 格式（user_id -> [(supplier_id, amt), ...]）
    allocations = {}
    total_flow_all = 0
    for user in users:
        uid = user["id"]
        allocs = []
        if uid in user_alloc_map:
            for sid, amt in user_alloc_map[uid].items():
                if amt > 0:
                    allocs.append((sid, int(amt)))
                    total_flow_all += int(amt)
        allocations[uid] = allocs

    # 最终偏好得分：greedy 的得分 + mcmf 计算得到的得分
    total_pref_all = int(greedy_pref_score + mcmf_pref_part)

    # 如果 reduce_neg_t 为 None，设为 0.0 便于展示
    if reduce_neg_t is None:
        reduce_neg_t = 0.0

    # timings 合并
    total_time = greedy_time + mcmf_time if (greedy_time is not None and mcmf_time is not None) else (greedy_time or mcmf_time or 0.0)

    result = {
        "total_flow": int(total_flow_all),
        "total_pref_score": int(total_pref_all),
        "allocations": allocations,
        "timings": {
            "greedy_time": float(greedy_time or 0.0),
            "mcmf_time": float(mcmf_time or 0.0),
            "total_time": float(total_time),
            "reduce_neg_cycle_time": float(reduce_neg_t)
        },
        # 备选信息（可选）：保留原始 greedy 与 mcmf 部分的拆分统计，方便调试或后续分析
        "breakdown": {
            "greedy_assigned": int(greedy_assigned),
            "greedy_pref_score": int(greedy_pref_score),
            "mcmf_assigned": int(mcmf_flow_part),
            "mcmf_pref_score": int(mcmf_pref_part)
        }
    }
    return result



if __name__ == "__main__":
    # 命令行示例：对单个实例运行
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inst", required=True, help="instance json path")
    parser.add_argument("--out", default=None, help="output json path")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--warm", action="store_true", help="disable warm-start")
    parser.add_argument("--max-fill", type=float, default=0.5, help="max_fill_fraction for partial greedy")
    parser.add_argument("--cancel-iter", type=int, default=200)
    parser.add_argument("--cancel-time", type=float, default=1.0)
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
