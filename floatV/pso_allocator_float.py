#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pso_allocator_float.py

PSO-based allocator, 支持整数实例与浮点实例（自动检测）。
改动摘要：
- 自动检测实例是否包含浮点 stock/need（容差 EPS）。
- 若为浮点实例，decode_particle_to_alloc 将执行“连续分配”策略（不再 floor），
  逐步按权重与供应剩余进行浮点分配，保证供应不超发且满足用户需求上限。
- 若为整数实例，保留原有 floor + fractional-fill 的整数化策略。
- evaluate_alloc 支持浮点数量的偏好评分（返回 float）。
- 输出 allocations 中分配量为 float（整数场景仍以 int 显示以兼容旧流程）。

Usage: 同原脚本接口
"""
import argparse
import json
import math
import random
import time
from collections import defaultdict
from copy import deepcopy

EPS = 1e-9

# ---------------------------
# 工具/repair/评估函数（已修改以支持浮点）
# ---------------------------

def build_candidates(inst, top_k=None):
    """
    为每个 user 构建候选供应商列表（按 score 降序），返回结构：
    {
      "suppliers": [ { "id": id, "stock": float }, ... ],
      "users": [ { "id": id, "need": float, "cands": [(sup_id, score), ...] }, ... ],
      "supplier_index": {sup_id: idx_in_suppliers}
    }
    """
    suppliers = deepcopy(inst.get("suppliers", []))
    users_raw = inst.get("users", [])
    supplier_index = {s["id"]: i for i, s in enumerate(suppliers)}

    users = []
    for u in users_raw:
        prefs = u.get("supplier_scores", [])
        prefs_sorted = sorted(prefs, key=lambda x: -float(x[1]))
        if top_k is not None:
            prefs_sorted = prefs_sorted[:top_k]
        prefs_sorted = [(sid, int(float(score))) for sid, score in prefs_sorted]
        # keep need/stock as float
        need_val = float(u.get("need", 0.0))
        users.append({"id": u["id"], "need": need_val, "cands": prefs_sorted})
    # ensure supplier stocks are floats
    for s in suppliers:
        s["stock"] = float(s.get("stock", 0.0))
    return {"suppliers": suppliers, "users": users, "supplier_index": supplier_index}

def is_integer_instance(inst):
    """
    检测实例是否为“整数实例”：所有 stock 与 need 接近整数（小于 EPS 的偏差视为整数）
    """
    for s in inst.get("suppliers", []):
        v = float(s.get("stock", 0.0))
        if abs(v - round(v)) > EPS:
            return False
    for u in inst.get("users", []):
        v = float(u.get("need", 0.0))
        if abs(v - round(v)) > EPS:
            return False
    return True

def decode_particle_to_alloc(pos, users, suppliers_stock, integer_mode=True):
    """
    将粒子 position 解码为分配（支持 integer_mode 或 continuous_mode）。
    - pos: concatenated weights for each user (length = sum_k k_u)
    - users: list of {id, need (float), cands: [(sup_id, score), ...]}
    - suppliers_stock: dict sup_id -> stock (float)
    - integer_mode: if True use original floor+fraction integerization; otherwise do continuous float allocation.
    返回:
      allocations dict user_id -> [(sup_id, amt), ...] (amt float for continuous, int for integer)
      total_flow (float)
    注意：不会修改传入 suppliers_stock（内部使用拷贝 rem_stock）
    """
    # prepare
    alloc_map = {str(u["id"]): [] for u in users}
    rem_stock = {str(k): float(v) for k, v in suppliers_stock.items()}
    idx = 0
    desired_float = {}
    per_user_meta = {}

    # 1) compute desired continuous allocations per user based on normalized weights
    for u in users:
        uid = str(u["id"])
        need = float(u["need"])
        k = len(u["cands"])
        slice_vec = pos[idx: idx + k]
        idx += k
        s = sum(slice_vec)
        if s <= EPS:
            weights = [0.0] * k
        else:
            weights = [max(0.0, v) / s for v in slice_vec]
        per_user_meta[uid] = {"need": need, "k": k, "weights": weights, "cands": [(str(sid), int(score)) for sid, score in u["cands"]]}
        for j, (sid, score) in enumerate(u["cands"]):
            amt = weights[j] * need
            desired_float[(uid, str(sid))] = float(amt)

    assigned = defaultdict(float)  # (uid,sid) -> float assigned

    if integer_mode:
        # ORIGINAL integer strategy: floor desired, then fractional-fill by priority
        # floor pass
        user_assigned_sum = defaultdict(int)
        for (uid, sid), d in desired_float.items():
            flo = int(math.floor(d + EPS))
            if flo <= 0:
                continue
            avail = int(math.floor(rem_stock.get(sid, 0.0) + EPS))
            take = min(flo, avail)
            if take > 0:
                assigned[(uid, sid)] += float(take)
                rem_stock[sid] = rem_stock.get(sid, 0.0) - float(take)
                user_assigned_sum[uid] += int(take)
        # fractional pass: build per-user fractional list sorted by fractional part and score
        frac_candidates = {}
        user_left = {}
        for uid, meta in per_user_meta.items():
            lst = []
            for sid, score in meta["cands"]:
                d = desired_float.get((uid, sid), 0.0)
                frac = d - math.floor(d + EPS)
                lst.append((frac, sid, score, d))
            lst_sorted = sorted(lst, key=lambda x: (-x[0], -x[2]))
            frac_candidates[uid] = lst_sorted
            user_left[uid] = int(max(0, meta["need"] - user_assigned_sum.get(uid, 0)))
        # assign leftover units per user by priority until supplier exhausted or user satisfied
        for uid in user_left:
            left = user_left[uid]
            if left <= 0:
                continue
            for frac, sid, score, d in frac_candidates[uid]:
                if left <= 0:
                    break
                avail = int(math.floor(rem_stock.get(sid, 0.0) + EPS))
                if avail <= 0:
                    continue
                take = min(left, avail)
                assigned[(uid, sid)] += float(take)
                rem_stock[sid] = rem_stock.get(sid, 0.0) - float(take)
                left -= int(take)
            # leftover remains unmet
        # build alloc_map with int amounts
        total_flow = 0
        for (uid, sid), amt in assigned.items():
            if amt >= 1 - EPS:
                q = int(round(amt))
                if q > 0:
                    alloc_map[uid].append((sid, q))
                    total_flow += q
        return alloc_map, float(total_flow)

    else:
        # CONTINUOUS FLOAT ALLOCATION STRATEGY
        # For each user, attempt to allocate up to need across its candidates, respecting rem_stock.
        for uid, meta in per_user_meta.items():
            need = meta["need"]
            # copy candidate lists and weights
            cands = [sid for sid, _ in meta["cands"]]
            weights = meta["weights"][:]  # same order as cands
            # if all weights zero, skip
            if sum(weights) <= EPS or need <= EPS:
                continue
            # track remaining need for this user
            rem_need = need
            # active candidate indices
            active = [i for i, sid in enumerate(cands) if rem_stock.get(cands[i], 0.0) > EPS]
            # if none active, skip
            if not active:
                continue
            # iterative proportional allocation: on each round allocate target = weight/sum(weights_active) * rem_need,
            # but clipped by supplier availability; update rem_need and active set until done or no progress
            max_rounds = len(active) + 5
            rounds = 0
            while rem_need > EPS and active and rounds < 1000:
                rounds += 1
                total_w = sum(weights[i] for i in active)
                if total_w <= EPS:
                    break
                progress = False
                for i in list(active):
                    sid = cands[i]
                    if rem_stock.get(sid, 0.0) <= EPS:
                        # remove exhausted
                        active.remove(i)
                        continue
                    target = (weights[i] / total_w) * rem_need
                    give = min(target, rem_stock.get(sid, 0.0))
                    if give > EPS:
                        assigned[(uid, sid)] += float(give)
                        rem_stock[sid] = rem_stock.get(sid, 0.0) - give
                        rem_need -= give
                        progress = True
                        if rem_stock[sid] <= EPS:
                            # mark removed in next iteration; remove by index value
                            try:
                                active.remove(i)
                            except ValueError:
                                pass
                if not progress:
                    # can't allocate more (suppliers exhausted or tiny remainders), break
                    break
            # if rem_need still > EPS, user remains partially unmet
        # build alloc_map with float amounts (filter tiny)
        total_flow = 0.0
        for (uid, sid), amt in assigned.items():
            if amt > EPS:
                alloc_map[uid].append((sid, float(amt)))
                total_flow += float(amt)
        return alloc_map, float(total_flow)

def evaluate_alloc(alloc_map, users_score_map):
    """
    计算总偏好分数（支持浮点量）
    returns (total_flow (float), total_pref_score (float))
    """
    total_flow = 0.0
    total_pref = 0.0
    for uid, lst in alloc_map.items():
        for sid, amt in lst:
            a = float(amt)
            total_flow += a
            score = users_score_map.get((str(uid), str(sid)))
            if score is None:
                continue
            total_pref += a * float(score)
    return float(total_flow), float(total_pref)


# ---------------------------
# PSO 主流程（保持原框架，但支持 float）
# ---------------------------

def run_pso(inst, top_k=10, particles=50, iters=200, w_inertia=0.6, c1=1.5, c2=1.5, seed=None, verbose=False):
    t0 = time.time()
    if seed is not None:
        random.seed(seed)

    data = build_candidates(inst, top_k=top_k)
    users = data["users"]
    suppliers = data["suppliers"]

    # suppliers_stock map sup_id -> stock (float)
    suppliers_stock = {str(s["id"]): float(s.get("stock", 0.0)) for s in suppliers}

    # detect integer vs float instance
    integer_mode = is_integer_instance(inst)

    # build dim
    user_slices = []
    dim = 0
    for u in users:
        k = len(u["cands"])
        user_slices.append((str(u["id"]), dim, k))
        dim += k

    if dim == 0:
        res = {"total_flow": 0.0, "total_pref_score": 0.0, "allocations": {u["id"]: [] for u in users}, "timings": {"pso_time": 0.0}}
        return res

    users_score_map = {}
    for u in users:
        uid = str(u["id"])
        for sid, score in u["cands"]:
            users_score_map[(uid, str(sid))] = int(score)

    # initialize particles
    pos = [[random.random() for _ in range(dim)] for _ in range(particles)]
    vel = [[(random.random() - 0.5) * 0.1 for _ in range(dim)] for _ in range(particles)]
    pbest_pos = [p[:] for p in pos]
    pbest_val = [None for _ in range(particles)]
    gbest_pos = None
    gbest_val = None

    # evaluate initial
    for i in range(particles):
        alloc_map, flow = decode_particle_to_alloc(pos[i], users, suppliers_stock, integer_mode=integer_mode)
        flow2, pref = evaluate_alloc(alloc_map, users_score_map)
        pbest_val[i] = pref
        pbest_pos[i] = pos[i][:]
        if gbest_val is None or pref > gbest_val:
            gbest_val = pref
            gbest_pos = pos[i][:]

    if verbose:
        print(f"[pso] init particles={particles}, dim={dim}, top_k={top_k}, integer_mode={integer_mode}, best_pref={gbest_val}")

    # main loop
    for it in range(iters):
        for i in range(particles):
            for d in range(dim):
                r1 = random.random()
                r2 = random.random()
                vel[i][d] = w_inertia * vel[i][d] + c1 * r1 * (pbest_pos[i][d] - pos[i][d]) + c2 * r2 * (gbest_pos[d] - pos[i][d])
                pos[i][d] += vel[i][d]
                # clamp to [0,1]
                if pos[i][d] < 0.0:
                    pos[i][d] = 0.0
                elif pos[i][d] > 1.0:
                    pos[i][d] = 1.0
            alloc_map, flow = decode_particle_to_alloc(pos[i], users, suppliers_stock, integer_mode=integer_mode)
            _, pref = evaluate_alloc(alloc_map, users_score_map)
            if pref is not None and (pbest_val[i] is None or pref > pbest_val[i]):
                pbest_val[i] = pref
                pbest_pos[i] = pos[i][:]
            if pref is not None and (gbest_val is None or pref > gbest_val):
                gbest_val = pref
                gbest_pos = pos[i][:]

        if verbose and (it % max(1, iters // 10) == 0):
            print(f"[pso] iter {it+1}/{iters}, best_pref={gbest_val}")

    alloc_map_best, flow_best = decode_particle_to_alloc(gbest_pos, users, suppliers_stock, integer_mode=integer_mode)
    flow_final, pref_final = evaluate_alloc(alloc_map_best, users_score_map)
    t1 = time.time()
    # format allocations: convert sid keys as original ids and amounts properly typed
    # if integer_mode, cast amounts to int for readability
    allocs_out = {}
    for uid, lst in alloc_map_best.items():
        outlst = []
        for sid, amt in lst:
            if integer_mode:
                outlst.append((str(sid), int(round(amt))))
            else:
                outlst.append((str(sid), float(amt)))
        allocs_out[str(uid)] = outlst

    res = {
        "total_flow": float(flow_final),
        "total_pref_score": float(pref_final),
        "allocations": allocs_out,
        "timings": {"pso_time": t1 - t0, "integer_mode": integer_mode}
    }
    return res

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="PSO-based allocator (supports float & integer instances)")
    parser.add_argument("--inst", required=True, help="instance json path")
    parser.add_argument("--out", default=None, help="output json path")
    parser.add_argument("--topk", type=int, default=None, help="per-user top-k suppliers to consider (default None -> all)")
    parser.add_argument("--particles", type=int, default=50, help="number of particles")
    parser.add_argument("--iters", type=int, default=200, help="PSO iterations")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--verbose", action="store_true", help="verbose")
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
