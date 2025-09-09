#!/usr/bin/env python3
"""
gen_instance.py

Generate synthetic supplier-user instance for MCMF experiments.

Output JSON structure:
{
  "suppliers":[ {"id":"s0","stock":int}, ... ],
  "users":[ {"id":"u0","need":int,"supplier_scores":[["s0",score], ...] }, ... ],
  "meta": { ... }
}

Each user will be connected to `avg_degree` random suppliers.
Scores can follow modes: 'uniform', 'skewed', 'clustered'.
"""
import argparse, json, random, math

def gen_scores(mode, num_suppliers):
    if mode == "uniform":
        return [random.randint(0,100) for _ in range(num_suppliers)]
    if mode == "skewed":
        # many small, few large: sample power-law-ish
        return [int(random.paretovariate(1.5) * 10) for _ in range(num_suppliers)]
    if mode == "clustered":
        # create a couple of clusters
        k = max(2, int(math.sqrt(num_suppliers)))
        centers = [random.randint(0,100) for _ in range(k)]
        res = []
        for _ in range(num_suppliers):
            c = random.choice(centers)
            res.append(max(0, min(100, int(random.gauss(c, 15)))))
        return res
    return [random.randint(0,100) for _ in range(num_suppliers)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--S", type=int, default=50, help="num suppliers")
    p.add_argument("--U", type=int, default=100, help="num users")
    p.add_argument("--avg_degree", type=int, default=5, help="avg edges per user")
    p.add_argument("--supply_min", type=int, default=10)
    p.add_argument("--supply_max", type=int, default=100)
    p.add_argument("--demand_min", type=int, default=1)
    p.add_argument("--demand_max", type=int, default=50)
    p.add_argument("--score_mode", choices=['uniform','skewed','clustered'], default='uniform')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="instances/test.json")
    args = p.parse_args()

    random.seed(args.seed)
    suppliers = []
    for i in range(args.S):
        suppliers.append({"id": f"s{i}", "stock": random.randint(args.supply_min, args.supply_max)})

    users = []
    for j in range(args.U):
        need = random.randint(args.demand_min, args.demand_max)
        # choose avg_degree distinct suppliers
        deg = min(args.avg_degree, args.S)
        sup_ids = random.sample(range(args.S), deg)
        scores = gen_scores(args.score_mode, args.S)
        supplier_scores = []
        for sid in sup_ids:
            supplier_scores.append([f"s{sid}", int(scores[sid])])
        users.append({"id": f"u{j}", "need": need, "supplier_scores": supplier_scores})

    inst = {"suppliers": suppliers, "users": users,
            "meta": {"S": args.S, "U": args.U, "avg_degree": args.avg_degree, "seed": args.seed, "score_mode": args.score_mode}}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(inst, f, indent=2)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
