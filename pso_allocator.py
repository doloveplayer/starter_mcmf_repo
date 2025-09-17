#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pso_allocator.py

基于粒子群（PSO）的资源分配实现。

设计：
- 将每个用户的候选供应商限制为 top_k（按 score 降序）。
- 粒子表示为：对每个用户 u，维护一个长度 k_u 的连续权重向量 w_u;
  解码时按权重将用户需求分配到候选供应商，最终通过整数化（先 floor，再按 fractional 部分/score 填充）
  并严格遵守 supplier 的 stock 上限与 user 的 need 上限。

输出（dict）:
{
  "total_flow": int,
  "total_pref_score": int,
  "allocations": { user_id: [(supplier_id, amt), ...], ... },
  "timings": {"pso_time": float}
}

python pso_allocator.py --inst instances/test_S160_U200_DEG60.json --out temp_pso.json --topk 10 --particles 80 --iters 400 --seed 123 --verbose
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from copy import deepcopy

# ---------------------------
# 工具/repair/评估函数
# ---------------------------

def build_candidates(inst, top_k=None):
    """
    为每个 user 构建候选供应商列表（按 score 降序），返回结构：
    {
      "suppliers": [ { "id": id, "stock": int }, ... ],
      "users": [ { "id": id, "need": int, "cands": [(sup_id, score), ...] }, ... ],
      "supplier_index": {sup_id: idx_in_suppliers}
    }
    保证每个用户的 cands 长度 <= top_k (若 top_k 为 None 则保留全部)。
    """
    suppliers = deepcopy(inst.get("suppliers", []))
    users_raw = inst.get("users", [])
    supplier_index = {s["id"]: i for i, s in enumerate(suppliers)}

    users = []
    for u in users_raw:
        prefs = u.get("supplier_scores", [])
        # convert score to int safely and sort desc
        prefs_sorted = sorted(prefs, key=lambda x: -int(x[1]))
        if top_k is not None:
            prefs_sorted = prefs_sorted[:top_k]
        # keep as list of tuples (sup_id, int(score))
        prefs_sorted = [(sid, int(score)) for sid, score in prefs_sorted]
        users.append({"id": u["id"], "need": int(u.get("need", 0)), "cands": prefs_sorted})
    return {"suppliers": suppliers, "users": users, "supplier_index": supplier_index}


def decode_particle_to_alloc(pos, users, suppliers_stock):
    """
    将粒子 position 解码为整数分配（repair），保证：
      - 对每个用户: sum allocated <= need
      - 对每个 supplier: sum allocated <= stock
    pos 是按用户串联的权重向量（每个用户的 k_u 个值，无需归一化）。
    users: list of {id, need, cands: [(sup_id, score), ...]}
    suppliers_stock: dict sup_id -> remaining stock (initial capacities)
    返回: allocations dict user_id -> list of (sup_id, amt), total_flow(int)
    说明：此函数会**消耗 suppliers_stock**拷贝，不会修改传入字典
    """
    # 构建 per-user slices
    alloc_map = {str(u["id"]): [] for u in users}
    # copy supplier remaining capacity
    rem_stock = {str(k): int(v) for k, v in suppliers_stock.items()}

    idx = 0
    # first pass: compute desired continuous allocations (float)
    # We'll do floor to integers, then second pass fill residuals
    desired_float = {}  # (user_id, sup_id) -> float
    per_user_meta = {}  # per user bookkeeping
    for u in users:
        uid = str(u["id"])
        need = int(u["need"])
        k = len(u["cands"])
        slice_vec = pos[idx: idx + k]
        idx += k

        # if all zeros -> no allocation
        s = sum(slice_vec)
        weights = []
        if s <= 0:
            weights = [0.0] * k
        else:
            weights = [max(0.0, v) / s for v in slice_vec]  # normalized nonnegative weights

        # desired continuous allocation
        for j, (sid, score) in enumerate(u["cands"]):
            amt = weights[j] * need
            desired_float[(uid, str(sid))] = float(amt)
        per_user_meta[uid] = {"need": need, "k": k, "cands": [(str(sid), int(score)) for sid, score in u["cands"]]}

    # second pass: floor and assign if supplier has capacity
    assigned = defaultdict(int)  # (uid,sid)->amt
    user_assigned_sum = defaultdict(int)
    for (uid, sid), d in desired_float.items():
        flo = math.floor(d)
        if flo <= 0:
            continue
        avail = rem_stock.get(sid, 0)
        take = min(flo, avail)
        if take > 0:
            assigned[(uid, sid)] += int(take)
            rem_stock[sid] = avail - take
            user_assigned_sum[uid] += int(take)

    # third pass: assign remaining units per user using fractional parts & scores
    # compute leftover need per user
    user_left = {}
    # also build fractional part priority
    frac_candidates = defaultdict(list)  # uid -> list of (priority, sid, score, frac_part)
    for (uid, sid), d in desired_float.items():
        need = per_user_meta[uid]["need"]
        cur_assigned = assigned.get((uid, sid), 0)
        frac = d - math.floor(d)
        frac_candidates[uid].append((frac, sid, per_user_meta[uid]["cands"][ [s[0] for s in per_user_meta[uid]["cands"] ].index(sid) ][1] if True else 0 , d) )
        # The above line with complicated indexing is awkward; instead we'll get score from cands mapping below.

    # rebuild frac_candidates more simply (and correctly)
    frac_candidates = {}
    for uid, meta in per_user_meta.items():
        lst = []
        for sid, score in meta["cands"]:
            d = desired_float.get((uid, sid), 0.0)
            frac = d - math.floor(d)
            lst.append((frac, sid, score, d))
        # sort by frac desc then score desc
        lst_sorted = sorted(lst, key=lambda x: (-x[0], -x[2]))
        frac_candidates[uid] = lst_sorted
        user_left[uid] = meta["need"] - user_assigned_sum.get(uid, 0)

    # Fill units one-by-one per user using frac order (tie-break by score), but ensure supplier capacity
    # Loop users, for each user while left>0 try to fill from its candidate list
    for uid in user_left:
        left = user_left[uid]
        if left <= 0:
            continue
        # attempt to allocate up to 'left' units
        # iterate candidates in priority order and assign as much as possible
        for frac, sid, score, d in frac_candidates[uid]:
            if left <= 0:
                break
            avail = rem_stock.get(sid, 0)
            if avail <= 0:
                continue
            # assign 1..left units (we can assign as many as avail but we favor fewer to be fair)
            take = min(left, avail)
            assigned[(uid, sid)] += int(take)
            rem_stock[sid] = avail - take
            left -= int(take)
        user_left[uid] = left

    # After above, some users may still have unmet need because suppliers exhausted.
    # We could attempt redistribution: for users with leftover need, try to take from other user's assigned
    # but that is complex. We'll accept partial fulfillment.

    # build alloc_map
    total_flow = 0
    for (uid, sid), amt in assigned.items():
        if amt > 0:
            alloc_map[str(uid)].append((int(sid) if str(sid).lstrip('-').isdigit() else sid, int(amt)))
            total_flow += int(amt)

    return alloc_map, total_flow


def evaluate_alloc(alloc_map, users_score_map):
    """
    计算总偏好分数 total_pref_score = sum_{user,sup} alloc * score
    users_score_map: dict (user_id, sup_id) -> score (int)
    返回 (total_flow, total_pref_score)
    """
    total_flow = 0
    total_pref = 0
    for uid, lst in alloc_map.items():
        for sid, amt in lst:
            total_flow += int(amt)
            score = users_score_map.get((str(uid), str(sid)))
            if score is None:
                # 如果没有 score，默认为 0
                continue
            total_pref += int(amt) * int(score)
    return int(total_flow), int(total_pref)


# ---------------------------
# PSO 主流程
# ---------------------------

def run_pso(inst, top_k=10, particles=50, iters=200, w_inertia=0.6, c1=1.5, c2=1.5, seed=None, verbose=False):
    """
    运行 PSO 优化器。
    返回与其他算法兼容的结果字典。
    """
    t0 = time.time()
    if seed is not None:
        random.seed(seed)

    # build candidates and helper maps
    data = build_candidates(inst, top_k=top_k)
    users = data["users"]
    suppliers = data["suppliers"]
    supplier_index = data["supplier_index"]

    # suppliers_stock map sup_id -> stock
    suppliers_stock = {str(s["id"]): int(s.get("stock", 0)) for s in suppliers}

    # build per-user slice lengths and total dimension
    user_slices = []  # list of (uid, start_idx, k)
    dim = 0
    for u in users:
        k = len(u["cands"])
        user_slices.append((str(u["id"]), dim, k))
        dim += k

    if dim == 0:
        # no edges at all
        res = {"total_flow": 0, "total_pref_score": 0, "allocations": {u["id"]: [] for u in users}, "timings": {"pso_time": 0.0}}
        return res

    # build score lookup (user_id, sup_id) -> score
    users_score_map = {}
    for u in users:
        uid = str(u["id"])
        for sid, score in u["cands"]:
            users_score_map[(uid, str(sid))] = int(score)

    # PSO variables: positions (list of list), velocities, pbest pos/value, gbest
    # positions initialized uniform random in [0,1)
    pos = [[random.random() for _ in range(dim)] for _ in range(particles)]
    vel = [[(random.random() - 0.5) * 0.1 for _ in range(dim)] for _ in range(particles)]  # small init vel
    pbest_pos = [p[:] for p in pos]
    pbest_val = [None for _ in range(particles)]
    gbest_pos = None
    gbest_val = None

    # evaluate initial particles
    for i in range(particles):
        alloc_map, flow = decode_particle_to_alloc(pos[i], users, suppliers_stock)
        flow2, pref = evaluate_alloc(alloc_map, users_score_map)
        # pref is objective to maximize
        pbest_val[i] = pref
        pbest_pos[i] = pos[i][:]
        if gbest_val is None or pref > gbest_val:
            gbest_val = pref
            gbest_pos = pos[i][:]

    if verbose:
        print(f"[pso] init particles={particles}, dim={dim}, top_k={top_k}, best_pref={gbest_val}")

    # main loop
    for it in range(iters):
        for i in range(particles):
            # update velocity and position
            for d in range(dim):
                r1 = random.random()
                r2 = random.random()
                vel[i][d] = w_inertia * vel[i][d] + c1 * r1 * (pbest_pos[i][d] - pos[i][d]) + c2 * r2 * (gbest_pos[d] - pos[i][d])
                pos[i][d] += vel[i][d]
                # clamp pos in [0, 1] (we treat negative as zero weight)
                if pos[i][d] < 0.0:
                    pos[i][d] = 0.0
                elif pos[i][d] > 1.0:
                    pos[i][d] = 1.0

            # evaluate
            alloc_map, flow = decode_particle_to_alloc(pos[i], users, suppliers_stock)
            flow2, pref = evaluate_alloc(alloc_map, users_score_map)

            # update pbest
            if pref is not None and (pbest_val[i] is None or pref > pbest_val[i]):
                pbest_val[i] = pref
                pbest_pos[i] = pos[i][:]
            # update gbest
            if pref is not None and (gbest_val is None or pref > gbest_val):
                gbest_val = pref
                gbest_pos = pos[i][:]

        if verbose and (it % max(1, iters // 10) == 0):
            print(f"[pso] iter {it+1}/{iters}, best_pref={gbest_val}")

    # decode best solution
    alloc_map_best, flow_best = decode_particle_to_alloc(gbest_pos, users, suppliers_stock)
    flow_final, pref_final = evaluate_alloc(alloc_map_best, users_score_map)

    t1 = time.time()
    res = {
        "total_flow": flow_final,
        "total_pref_score": pref_final,
        "allocations": {str(k): v for k, v in alloc_map_best.items()},
        "timings": {"pso_time": t1 - t0}
    }
    return res


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="PSO-based allocator for the supplier-user allocation problem")
    parser.add_argument("--inst", required=True, help="instance json path")
    parser.add_argument("--out", default=None, help="output json path")
    parser.add_argument("--topk", type=int, default=None, help="per-user top-k suppliers to consider (default 10)")
    parser.add_argument("--particles", type=int, default=50, help="number of PSO particles (default 50)")
    parser.add_argument("--iters", type=int, default=200, help="PSO iterations (default 200)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    args = parser.parse_args()

    with open(args.inst, "r", encoding="utf-8") as f:
        inst = json.load(f)

    res = run_pso(inst, top_k=args.topk, particles=args.particles, iters=args.iters, seed=args.seed, verbose=args.verbose)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            json.dump(res, fo, indent=2, ensure_ascii=False)
        if args.verbose:
            print("Wrote result to", args.out)
    else:
        import pprint
        pprint.pprint(res)


if __name__ == "__main__":
    main()
