#!/usr/bin/env python3
# File: gen_instances_bulk.py
# Usage example:
# python gen_instances_bulk.py --outdir instances_bulk --S 20 40 80 160 --U 50 100 200 400 --deg 5 20 50 --seeds 1 2 3 --score_mode clustered --supply_min 10 --supply_max 100 --demand_min 1 --demand_max 50
#
import argparse
import json
import os
import random
import math
from pathlib import Path

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
    # fallback
    return [random.randint(0,100) for _ in range(num_suppliers)]

def gen_instance(S, U, avg_degree, supply_min, supply_max, demand_min, demand_max, score_mode, seed):
    random.seed(seed)
    suppliers = []
    for i in range(S):
        suppliers.append({"id": f"s{i}", "stock": random.randint(supply_min, supply_max)})
    users = []
    scores_all = gen_scores(score_mode, S)
    for j in range(U):
        need = random.randint(demand_min, demand_max)
        deg = min(avg_degree, S)
        sup_ids = random.sample(range(S), deg)
        supplier_scores = []
        # map to chosen suppliers only, use scores_all for stable distribution
        for sid in sup_ids:
            supplier_scores.append([f"s{sid}", int(scores_all[sid])])
        users.append({"id": f"u{j}", "need": need, "supplier_scores": supplier_scores})
    inst = {"suppliers": suppliers, "users": users,
            "meta": {"S": S, "U": U, "avg_degree": avg_degree, "score_mode": score_mode, "seed": seed}}
    return inst

def main():
    p = argparse.ArgumentParser(description="Bulk generate MCMF instances over parameter grid")
    p.add_argument("--outdir", default="instances_bulk", help="directory to save instances")
    p.add_argument("--S", nargs="+", type=int, default=[20,40,80,160], help="supplier counts")
    p.add_argument("--U", nargs="+", type=int, default=[50,100,200,400], help="user counts")
    p.add_argument("--deg", nargs="+", type=int, default=[5,20,50], help="avg_degree values")
    p.add_argument("--seeds", nargs="+", type=int, default=[1,2,3], help="random seeds")
    p.add_argument("--score_mode", choices=['uniform','skewed','clustered'], default='clustered')
    p.add_argument("--supply_min", type=int, default=10)
    p.add_argument("--supply_max", type=int, default=100)
    p.add_argument("--demand_min", type=int, default=1)
    p.add_argument("--demand_max", type=int, default=50)
    p.add_argument("--max_instances", type=int, default=None, help="optional cap on total number generated")
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
        inst = gen_instance(S, U, deg, args.supply_min, args.supply_max, args.demand_min, args.demand_max, args.score_mode, seed)
        fname = f"inst_S{S}_U{U}_deg{deg}_seed{seed}_{args.score_mode}.json"
        fpath = outdir / fname
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(inst, f, indent=2)
        print("Wrote", fpath)

if __name__ == "__main__":
    main()
