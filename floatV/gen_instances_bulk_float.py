#!/usr/bin/env python3
# File: gen_instances_bulk_enhanced_float.py
"""
增强版实例生成脚本（支持小数库存/需求）
- 新增参数: --float (enable float stocks/needs), --precision (decimal places)
示例：
python gen_instances_bulk_enhanced_float.py --outdir instances_bulk --S 20 40 --U 50 100 --deg 5 20 --seeds 1 2 \
    --score_mode clustered --supply_demand_ratio 0.6 --pref_hot_fraction 0.3 --hot_suppliers 3 --pref_bias 30 \
    --pref_clustered --cluster_num 3 --float --precision 2
"""
import argparse
import json
import os
import random
import math
from pathlib import Path
from collections import defaultdict

def gen_scores(mode, num_suppliers):
    if mode == "uniform":
        return [random.randint(0,100) for _ in range(num_suppliers)]
    if mode == "skewed":
        # power-law-ish: many small, few large
        return [int(min(100, random.paretovariate(1.5) * 10)) for _ in range(num_suppliers)]
    if mode == "clustered":
        k = max(2, int(math.sqrt(num_suppliers)))
        centers = [random.randint(0,100) for _ in range(k)]
        res = []
        for _ in range(num_suppliers):
            c = random.choice(centers)
            res.append(max(0, min(100, int(random.gauss(c, 15)))))
        return res
    return [random.randint(0,100) for _ in range(num_suppliers)]

def allocate_supply_total(S, total_supply, supply_min, supply_max, alloc_mode="random", scale=1):
    """
    将 total_supply 分配到 S 个供应点，返回长度 S 的分配 (整数 if scale==1 else floats divided by scale)
    If scale > 1, all computations done on integers (total_supply, mins, maxs scaled) then results divided by scale.
    alloc_mode: "random" or "skewed"
    """
    assert S > 0
    # scale inputs to integer domain
    if scale > 1:
        total_supply_i = int(round(total_supply * scale))
        min_i = int(round(supply_min * scale)) if supply_min is not None else 0
        max_i = int(round(supply_max * scale)) if supply_max is not None else None
    else:
        total_supply_i = int(round(total_supply))
        min_i = int(supply_min) if supply_min is not None else 0
        max_i = int(supply_max) if supply_max is not None else None

    # generate weights
    if alloc_mode == "skewed":
        weights = [random.paretovariate(1.5) for _ in range(S)]
    else:
        weights = [random.random() + 0.01 for _ in range(S)]
    total_w = sum(weights)
    raw = [w / total_w * total_supply_i for w in weights]
    alloc_i = [int(math.floor(x)) for x in raw]
    rem = total_supply_i - sum(alloc_i)
    # distribute remainder by fractional parts
    frac = [(raw[i]-alloc_i[i], i) for i in range(S)]
    frac.sort(reverse=True)
    idx = 0
    while rem > 0 and idx < S:
        alloc_i[frac[idx][1]] += 1
        rem -= 1
        idx += 1
        if idx == S and rem > 0:
            idx = 0
    # enforce mins
    for i in range(S):
        if min_i is not None and alloc_i[i] < min_i:
            diff = min_i - alloc_i[i]
            alloc_i[i] = min_i
            j = 0
            while diff > 0 and j < S:
                if j != i and alloc_i[j] > min_i:
                    take = min(diff, alloc_i[j] - min_i)
                    alloc_i[j] -= take
                    diff -= take
                j += 1
    # enforce maxs
    for i in range(S):
        if max_i is not None and alloc_i[i] > max_i:
            diff = alloc_i[i] - max_i
            alloc_i[i] = max_i
            j = 0
            while diff > 0 and j < S:
                if j != i and alloc_i[j] < (max_i if max_i is not None else total_supply_i):
                    give = min(diff, (max_i if max_i is not None else total_supply_i) - alloc_i[j])
                    alloc_i[j] += give
                    diff -= give
                j += 1
    # final adjust if sum mismatch
    cur_sum = sum(alloc_i)
    if cur_sum != total_supply_i:
        diff = total_supply_i - cur_sum
        i = 0
        step = 1 if diff > 0 else -1
        while diff != 0:
            idx = i % S
            if step > 0:
                if max_i is None or alloc_i[idx] < max_i:
                    alloc_i[idx] += 1
                    diff -= 1
            else:
                if alloc_i[idx] > min_i:
                    alloc_i[idx] -= 1
                    diff -= 1
            i += 1
            if i > S * 1000:
                break

    # convert back to float if scale > 1
    if scale > 1:
        return [alloc_i[i] / scale for i in range(S)]
    else:
        return alloc_i

def add_concentrated_preferences(users, S, pref_hot_fraction, hot_suppliers, pref_bias, pref_clustered=False, cluster_num=1, seed=0):
    """
    修改 users 的 supplier_scores，使得一部分用户对若干 hot_suppliers 偏好上升（加 pref_bias）。
    If pref_clustered=True, split hot users into cluster_num groups, each group prefers a different small set of hot suppliers.
    """
    random.seed(seed + 12345)
    U = len(users)
    hot_user_count = int(round(pref_hot_fraction * U))
    if hot_user_count <= 0 or hot_suppliers <= 0:
        return users  # nothing to change

    all_supplier_ids = [f"s{i}" for i in range(S)]
    # choose hot suppliers set(s)
    if not pref_clustered or cluster_num <= 1:
        hot_set = random.sample(all_supplier_ids, min(hot_suppliers, S))
        hot_groups = [hot_set]
    else:
        hot_groups = []
        for c in range(cluster_num):
            k = max(1, min(hot_suppliers, S // cluster_num))
            group = random.sample(all_supplier_ids, k)
            hot_groups.append(group)

    hot_user_indices = random.sample(range(U), hot_user_count)
    if pref_clustered and cluster_num > 1:
        hot_user_indices.sort()
        groups_idx = []
        per = max(1, hot_user_count // cluster_num)
        for i in range(cluster_num):
            start = i * per
            end = start + per if i < cluster_num - 1 else hot_user_count
            groups_idx.append(hot_user_indices[start:end])
    else:
        groups_idx = [hot_user_indices]

    # apply bias
    if pref_clustered and cluster_num > 1:
        for gi, idx_list in enumerate(groups_idx):
            group_hot = hot_groups[gi % len(hot_groups)]
            for ui in idx_list:
                orig = users[ui]["supplier_scores"]
                d = {sid: int(sc) for sid, sc in orig}
                for sid in group_hot:
                    if sid in d:
                        d[sid] = min(100, d[sid] + pref_bias)
                    else:
                        d[sid] = min(100, pref_bias)
                new_list = []
                seen = set()
                for sid, sc in orig:
                    new_list.append([sid, d.get(sid, int(sc))])
                    seen.add(sid)
                for sid, sc in d.items():
                    if sid not in seen:
                        new_list.append([sid, sc])
                users[ui]["supplier_scores"] = new_list
    else:
        hot_set = hot_groups[0]
        for ui in hot_user_indices:
            orig = users[ui]["supplier_scores"]
            d = {sid: int(sc) for sid, sc in orig}
            for sid in hot_set:
                if sid in d:
                    d[sid] = min(100, d[sid] + pref_bias)
                else:
                    d[sid] = min(100, pref_bias)
            new_list = []
            seen = set()
            for sid, sc in orig:
                new_list.append([sid, d.get(sid, int(sc))])
                seen.add(sid)
            for sid, sc in d.items():
                if sid not in seen:
                    new_list.append([sid, sc])
            users[ui]["supplier_scores"] = new_list

    return users

def gen_instance(S, U, avg_degree, supply_min, supply_max, demand_min, demand_max, score_mode, seed,
                 supply_demand_ratio=None, supply_alloc_mode="random",
                 pref_hot_fraction=0.0, hot_suppliers=0, pref_bias=0, pref_clustered=False, cluster_num=1,
                 float_mode=False, precision=2):
    """
    生成一个实例，支持 supply/demand ratio 控制与集中偏好注入。
    如果 float_mode=True，会生成带小数的 stock/need，精度由 precision 控制（小数位）。
    Internally uses integer allocation on scaled values to preserve totals.
    """
    random.seed(seed)
    scale = 1
    if float_mode:
        scale = 10 ** max(0, int(precision))

    # 1) 先生成用户需求（浮点或整数）
    users = []
    needs = []
    for j in range(U):
        if float_mode:
            val = random.uniform(demand_min, demand_max)
            # round to precision
            need = round(val, precision)
        else:
            need = random.randint(demand_min, demand_max)
        needs.append(need)
        users.append({"id": f"u{j}", "need": need, "supplier_scores": []})

    total_demand = sum(needs)

    # 2) total_supply by ratio or random per-supplier
    if supply_demand_ratio is not None:
        total_supply = max(0.0, round(total_demand * float(supply_demand_ratio), precision if float_mode else 0))
        # ensure at least sum of mins if provided
        if supply_min is not None:
            min_sum = (supply_min * S) if not float_mode else (supply_min * S)
            if min_sum > total_supply:
                total_supply = min_sum
    else:
        total_supply = None

    # 3) allocate supplier stocks
    suppliers = []
    if total_supply is not None:
        # adapt supply_min/supply_max to floats if float_mode
        smin = supply_min if supply_min is not None else None
        smax = supply_max if supply_max is not None else None
        stocks = allocate_supply_total(S, total_supply, smin, smax, alloc_mode=supply_alloc_mode, scale=scale)
        for i in range(S):
            val = float(stocks[i]) if float_mode else int(stocks[i])
            suppliers.append({"id": f"s{i}", "stock": val})
    else:
        for i in range(S):
            if float_mode:
                val = round(random.uniform(supply_min, supply_max), precision)
            else:
                val = random.randint(supply_min, supply_max)
            suppliers.append({"id": f"s{i}", "stock": val})

    # 4) base scores
    base_scores = gen_scores(score_mode, S)

    # 5) user candidate suppliers with scores
    for j in range(U):
        deg = min(avg_degree, S)
        sup_ids = random.sample(range(S), deg)
        supplier_scores = []
        for sid in sup_ids:
            supplier_scores.append([f"s{sid}", int(base_scores[sid])])
        users[j]["supplier_scores"] = supplier_scores

    # 6) concentrated preferences
    if pref_hot_fraction > 0 and hot_suppliers > 0 and pref_bias != 0:
        users = add_concentrated_preferences(users, S, pref_hot_fraction, hot_suppliers, pref_bias,
                                            pref_clustered=pref_clustered, cluster_num=cluster_num, seed=seed)

    inst = {"suppliers": suppliers, "users": users,
            "meta": {"S": S, "U": U, "avg_degree": avg_degree, "score_mode": score_mode,
                     "seed": seed, "supply_demand_ratio": supply_demand_ratio,
                     "pref_hot_fraction": pref_hot_fraction, "hot_suppliers": hot_suppliers,
                     "pref_bias": pref_bias, "pref_clustered": pref_clustered, "cluster_num": cluster_num,
                     "float_mode": bool(float_mode), "precision": int(precision)}}
    return inst

def main():
    p = argparse.ArgumentParser(description="Bulk generate MCMF instances (enhanced, float support)")
    p.add_argument("--outdir", default="instances_bulk", help="directory to save instances")
    p.add_argument("--S", nargs="+", type=int, default=[20,40,80,160], help="supplier counts")
    p.add_argument("--U", nargs="+", type=int, default=[50,100,200,400], help="user counts")
    p.add_argument("--deg", nargs="+", type=int, default=[5,20,50], help="avg_degree values")
    p.add_argument("--seeds", nargs="+", type=int, default=[1,2,3], help="random seeds")
    p.add_argument("--score_mode", choices=['uniform','skewed','clustered'], default='clustered')
    p.add_argument("--supply_min", type=float, default=10.0)
    p.add_argument("--supply_max", type=float, default=100.0)
    p.add_argument("--demand_min", type=float, default=1.0)
    p.add_argument("--demand_max", type=float, default=50.0)
    p.add_argument("--max_instances", type=int, default=None, help="optional cap on total number generated")
    # new args
    p.add_argument("--supply_demand_ratio", type=float, default=None,
                   help="if set, total_supply = round(total_demand * ratio)")
    p.add_argument("--supply_alloc_mode", choices=['random','skewed'], default='random',
                   help="how to allocate total supply across suppliers when ratio used")
    p.add_argument("--pref_hot_fraction", type=float, default=0.0,
                   help="fraction of users that favor hot suppliers (0..1)")
    p.add_argument("--hot_suppliers", type=int, default=0, help="number of hot suppliers")
    p.add_argument("--pref_bias", type=int, default=0, help="additive bias to hot suppliers' scores (0..100)")
    p.add_argument("--pref_clustered", action="store_true", help="enable clustered concentrated preferences")
    p.add_argument("--cluster_num", type=int, default=1, help="number of clusters when pref_clustered set")
    # float options
    p.add_argument("--float", dest="float_mode", action="store_true", help="generate float stocks/needs (with precision)")
    p.add_argument("--precision", type=int, default=2, help="decimal places when --float is set")

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    combos = []
    for S in args.S:
        for U in args.U:
            for deg in args.deg:
                for seed in args.seeds:
                    combos.append((S, U, deg, seed))
    if args.max_instances is not None:
        combos = combos[:args.max_instances]

    print(f"Generating {len(combos)} instances to {outdir} ...")
    for S, U, deg, seed in combos:
        inst = gen_instance(S, U, deg, args.supply_min, args.supply_max, args.demand_min, args.demand_max,
                            args.score_mode, seed,
                            supply_demand_ratio=args.supply_demand_ratio,
                            supply_alloc_mode=args.supply_alloc_mode,
                            pref_hot_fraction=args.pref_hot_fraction,
                            hot_suppliers=args.hot_suppliers,
                            pref_bias=args.pref_bias,
                            pref_clustered=args.pref_clustered,
                            cluster_num=args.cluster_num,
                            float_mode=args.float_mode,
                            precision=args.precision)
        fname = f"inst_S{S}_U{U}_deg{deg}_seed{seed}_{args.score_mode}"
        if args.supply_demand_ratio is not None:
            fname += f"_sdr{args.supply_demand_ratio}"
        if args.pref_hot_fraction and args.hot_suppliers:
            fname += f"_hot{args.hot_suppliers}_pf{args.pref_hot_fraction}_bias{args.pref_bias}"
            if args.pref_clustered:
                fname += f"_clusters{args.cluster_num}"
        if args.float_mode:
            fname += f"_float_p{args.precision}"
        fname += ".json"
        fpath = outdir / fname
        # Use ensure_ascii=False and default float formatting
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(inst, f, indent=2, ensure_ascii=False)
        print("Wrote", fpath)

if __name__ == "__main__":
    main()
