#!/usr/bin/env python3
"""
gen_instances_controlled.py

精简版实例生成器 — 支持场景控制（稀缺、充足、自定义）与偏好分布选择。

示例:
python gen_instances_controlled.py --outdir instances --S 20 50 --U 100 --deg 5 --seeds 1 2 \
    --scenario scarce --score_dist clustered --pref_hot_fraction 0.2 --hot_suppliers 3 --pref_bias 30

场景:
 - scarce   -> 默认 ratio = 0.3 (供应远小于需求)
 - abundant -> 默认 ratio = 1.3 (供应略大于需求)
 - balanced -> 默认 ratio = 1.0
 - custom   -> 使用 --supply_demand_ratio 指定
 - none     -> 不使用比值，按每个供应点随机 stock（fallback）
"""
import argparse, json, random, math
from pathlib import Path

def gen_scores(dist, S, seed=0):
    random.seed(seed + 9876)
    if dist == "random":
        return [random.randint(0,100) for _ in range(S)]
    if dist == "skewed":
        return [min(100, int(random.paretovariate(1.7) * 8)) for _ in range(S)]
    if dist == "clustered":
        k = max(2, int(math.sqrt(S)))
        centers = [random.randint(0,100) for _ in range(k)]
        res = [max(0, min(100, int(random.gauss(random.choice(centers), 12)))) for _ in range(S)]
        return res
    # fallback
    return [random.randint(0,100) for _ in range(S)]

def allocate_total_supply(S, total_supply, supply_min, supply_max, mode="random", seed=0):
    """
    将 total_supply 分配到 S 个供应点（整数列表）。
    尽量遵守 min/max；若冲突以总量为先并做循环调整。
    """
    random.seed(seed + 123)
    if S == 0:
        return []
    # 权重：random 或 skewed
    if mode == "skewed":
        weights = [random.paretovariate(1.6) for _ in range(S)]
    else:
        weights = [random.random() + 0.01 for _ in range(S)]
    total_w = sum(weights)
    raw = [w / total_w * total_supply for w in weights]
    alloc = [int(math.floor(x)) for x in raw]
    # distribute remainder by fractional parts
    rem = total_supply - sum(alloc)
    frac = sorted([(raw[i]-alloc[i], i) for i in range(S)], reverse=True)
    i = 0
    while rem > 0:
        alloc[frac[i % S][1]] += 1
        rem -= 1
        i += 1
    # enforce min/max by simple clamp & re-balance
    # clamp mins
    for i in range(S):
        if supply_min is not None and alloc[i] < supply_min:
            alloc[i] = supply_min
    # clamp maxes and collect overflow/underflow
    overflow = 0
    for i in range(S):
        if supply_max is not None and alloc[i] > supply_max:
            overflow += alloc[i] - supply_max
            alloc[i] = supply_max
    # if overflow > 0, try to distribute to slots < max
    if overflow > 0:
        for i in range(S):
            if overflow == 0:
                break
            cap = (supply_max or total_supply) - alloc[i]
            if cap > 0:
                give = min(cap, overflow)
                alloc[i] += give
                overflow -= give
    # final adjust to match total_supply
    cur = sum(alloc)
    diff = total_supply - cur
    idx = 0
    while diff != 0:
        j = idx % S
        if diff > 0:
            if supply_max is None or alloc[j] < supply_max:
                alloc[j] += 1
                diff -= 1
        else:
            if supply_min is None or alloc[j] > supply_min:
                alloc[j] -= 1
                diff += 1
        idx += 1
        if idx > S * 1000:
            break
    return alloc

import random, math

def inject_hot_preferences(users, S, hot_fraction, hot_suppliers, bias,
                           clustered=False, cluster_num=1, seed=0,
                           insert_frac=0.3, min_insert=1):
    """
    对一部分用户（hot_fraction）提高对若干热供应点的偏好（加 bias）。
    插入规则：对于每个热用户，从对应的热供应点集合中随机插入 k 个，
    其中 k ∈ [min_insert, max_insert]，max_insert = max(1, ceil(insert_frac * len(group_hot)))
    - 若候选列表已有该供应点，则仅增加分数（不重复插入）。
    - insert_frac 默认为 0.3（即最多 30%）。
    """
    if hot_fraction <= 0 or hot_suppliers <= 0 or bias == 0:
        return users

    random.seed(seed + 5555)
    U = len(users)
    hot_count = max(1, int(round(hot_fraction * U)))

    supplier_ids = [f"s{i}" for i in range(S)]

    # prepare hot groups (clustered or global)
    if clustered and cluster_num > 1:
        hot_groups = []
        for _ in range(cluster_num):
            k = max(1, min(hot_suppliers, max(1, S // cluster_num)))
            hot_groups.append(random.sample(supplier_ids, k))
    else:
        hot_groups = [random.sample(supplier_ids, min(hot_suppliers, S))]

    # pick hot users indices
    hot_users = random.sample(range(U), hot_count)

    # if clustered: split hot_users into groups (round-robin assignment)
    if clustered and cluster_num > 1:
        # assign each hot user to a group index in round-robin fashion
        assignments = {hot_users[i]: i % len(hot_groups) for i in range(len(hot_users))}
    else:
        assignments = {ui: 0 for ui in hot_users}

    for ui, group_idx in assignments.items():
        group_hot = hot_groups[group_idx]
        L = len(group_hot)
        # compute max insert count: at most ceil(insert_frac * L), but at least min_insert and at most L
        max_insert = max(1, int(math.ceil(insert_frac * L)))
        max_insert = min(max_insert, L)
        # choose actual number to insert (random between min_insert and max_insert)
        num_to_insert = random.randint(min_insert, max_insert)
        # select that many hot suppliers
        chosen_hot = set(random.sample(group_hot, num_to_insert))

        # convert existing supplier_scores to dict for easy update
        orig = users[ui]["supplier_scores"]
        d = {sid: int(sc) for sid, sc in orig}

        # apply bias to chosen_hot only
        for sid in chosen_hot:
            d[sid] = min(100, d.get(sid, 0) + bias)

        # reconstruct list: preserve original ordering, append new ones at end
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


def gen_instance(S, U, avg_deg, supply_min, supply_max, demand_min, demand_max,
                 score_dist, seed, supply_demand_ratio=None, supply_alloc_mode="random",
                 pref_hot_fraction=0.0, hot_suppliers=0, pref_bias=0, pref_clustered=False, cluster_num=1):
    random.seed(seed)
    # users and needs
    users = []
    needs = [random.randint(demand_min, demand_max) for _ in range(U)]
    for j in range(U):
        users.append({"id": f"u{j}", "need": needs[j], "supplier_scores": []})
    total_demand = sum(needs)

    # supply total
    total_supply = None
    if supply_demand_ratio is not None:
        total_supply = max(0, int(round(total_demand * supply_demand_ratio)))
        # ensure min_sum
        if supply_min is not None:
            min_sum = supply_min * S
            if min_sum > total_supply:
                total_supply = min_sum

    # generate supplier stocks
    suppliers = []
    if total_supply is None:
        # per-supplier random stocks
        for i in range(S):
            st = random.randint(supply_min, supply_max)
            suppliers.append({"id": f"s{i}", "stock": st})
    else:
        stocks = allocate_total_supply(S, total_supply, supply_min, supply_max, mode=supply_alloc_mode, seed=seed)
        for i, st in enumerate(stocks):
            suppliers.append({"id": f"s{i}", "stock": int(st)})

    # base scores for all suppliers
    base_scores = gen_scores(score_dist, S, seed=seed)

    # each user samples candidate suppliers (deg)
    for j in range(U):
        deg = min(avg_deg, S)
        cand = random.sample(range(S), deg)
        users[j]["supplier_scores"] = [[f"s{sid}", int(base_scores[sid])] for sid in cand]

    # inject concentrated/hot preferences if requested
    if pref_hot_fraction > 0 and hot_suppliers > 0 and pref_bias != 0:
        users = inject_hot_preferences(users, S, pref_hot_fraction, hot_suppliers,
                                       pref_bias, clustered=pref_clustered, cluster_num=cluster_num, seed=seed)

    meta = {"S": S, "U": U, "avg_deg": avg_deg, "score_dist": score_dist, "seed": seed,
            "supply_demand_ratio": supply_demand_ratio, "supply_alloc_mode": supply_alloc_mode,
            "pref_hot_fraction": pref_hot_fraction, "hot_suppliers": hot_suppliers,
            "pref_bias": pref_bias, "pref_clustered": pref_clustered, "cluster_num": cluster_num}
    return {"suppliers": suppliers, "users": users, "meta": meta}

def main():
    p = argparse.ArgumentParser(description="Generate controlled MCMF instances")
    p.add_argument("--outdir", default="instances", help="output directory")
    p.add_argument("--S", nargs="+", type=int, default=[20], help="supplier counts")
    p.add_argument("--U", nargs="+", type=int, default=[100], help="user counts")
    p.add_argument("--deg", nargs="+", type=int, default=[5], help="avg degree(s)")
    p.add_argument("--seeds", nargs="+", type=int, default=[1], help="random seeds")
    p.add_argument("--score_mode", choices=["random","clustered","skewed"], default="clustered",
                   help="preference score distribution")
    p.add_argument("--supply_min", type=int, default=5)
    p.add_argument("--supply_max", type=int, default=100)
    p.add_argument("--demand_min", type=int, default=1)
    p.add_argument("--demand_max", type=int, default=50)
    p.add_argument("--scenario", choices=["scarce","abundant","balanced","custom","none"], default="none",
                   help="predefined supply/demand scenarios")
    p.add_argument("--supply_demand_ratio", type=float, default=None, help="use when scenario=custom or to override")
    p.add_argument("--supply_alloc_mode", choices=["random","skewed"], default="random")
    p.add_argument("--pref_hot_fraction", type=float, default=0.0)
    p.add_argument("--hot_suppliers", type=int, default=0)
    p.add_argument("--pref_bias", type=int, default=0)
    p.add_argument("--pref_clustered", action="store_true")
    p.add_argument("--cluster_num", type=int, default=1)
    p.add_argument("--max_instances", type=int, default=None)
    args = p.parse_args()

    # map scenarios to default ratios
    scenario_ratios = {"scarce": 0.4, "abundant": 1.2, "balanced": 1.0}
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    combos = []
    for S in args.S:
        for U in args.U:
            for deg in args.deg:
                for seed in args.seeds:
                    combos.append((S,U,deg,seed))
    if args.max_instances is not None:
        combos = combos[:args.max_instances]

    print(f"Generating {len(combos)} instances -> {outdir}")
    for S,U,deg,seed in combos:
        # decide ratio
        ratio = None
        if args.scenario in scenario_ratios:
            ratio = scenario_ratios[args.scenario]
        elif args.scenario == "custom":
            ratio = args.supply_demand_ratio
        elif args.scenario == "none":
            ratio = args.supply_demand_ratio  # may be None -> per-supplier random

        inst = gen_instance(S, U, deg, args.supply_min, args.supply_max, args.demand_min, args.demand_max,
                            args.score_mode, seed,
                            supply_demand_ratio=ratio,
                            supply_alloc_mode=args.supply_alloc_mode,
                            pref_hot_fraction=args.pref_hot_fraction,
                            hot_suppliers=args.hot_suppliers,
                            pref_bias=args.pref_bias,
                            pref_clustered=args.pref_clustered,
                            cluster_num=args.cluster_num)

        # descriptive filename
        fname = f"inst_S{S}_U{U}_deg{deg}_seed{seed}_{args.score_mode}"
        if ratio is not None:
            fname += f"_sdr{ratio}"
        if args.pref_hot_fraction and args.hot_suppliers:
            fname += f"_hot{args.hot_suppliers}_pf{args.pref_hot_fraction}_bias{args.pref_bias}"
            if args.pref_clustered:
                fname += f"_clusters{args.cluster_num}"
        fname += ".json"
        fpath = outdir / fname
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(inst, f, indent=2, ensure_ascii=False)
        print("Wrote", fpath)

if __name__ == "__main__":
    main()
