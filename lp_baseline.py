#!/usr/bin/env python3

def run_lp_on_instance(inst):
    try:
        import pulp
    except Exception as e:
        raise ImportError("PuLP is required for LP baseline. Install with `pip install pulp`. Error: " + str(e))
    suppliers = inst["suppliers"]
    users = inst["users"]
    # Build index maps
    sup_idx = {sup["id"]: i for i, sup in enumerate(suppliers)}
    user_idx = {u["id"]: j for j, u in enumerate(users)}

    # Collect edges
    edges = []
    for user in users:
        for sid, score in user.get("supplier_scores", []):
            if sid in sup_idx:
                edges.append((sid, user["id"], int(score)))

    # Create LP problem
    prob = pulp.LpProblem("lp_baseline", pulp.LpMinimize)
    var = {}
    for sid, uid, score in edges:
        name = f"f_{sid}_{uid}"
        var[(sid, uid)] = pulp.LpVariable(name, lowBound=0, cat='Continuous')

    # Objective: minimize sum(-score * f)
    prob += pulp.lpSum([(-score) * var[(sid, uid)] for sid, uid, score in edges])

    # Supplier capacity constraints
    for sup in suppliers:
        sid = sup["id"]
        cap = float(sup["stock"])
        relevant = [v for (s, u), v in var.items() if s == sid]
        if relevant:
            prob += pulp.lpSum(relevant) <= cap, f"sup_cap_{sid}"

    # User demand constraints
    for user in users:
        uid = user["id"]
        need = float(user["need"])
        relevant = [v for (s, u), v in var.items() if u == uid]
        if relevant:
            prob += pulp.lpSum(relevant) <= need, f"user_need_{uid}"

    # Solve using CBC if available
    try:
        solver = pulp.PULP_CBC_CMD(msg=False)
    except Exception:
        solver = None

    prob.solve(solver)

    status = pulp.LpStatus.get(prob.status, str(prob.status))
    allocations = {u["id"]: [] for u in users}
    total_pref = 0.0
    total_assigned = 0.0
    # score lookup
    score_map = {}
    for user in users:
        for sid, score in user.get("supplier_scores", []):
            score_map[(sid, user["id"])] = int(score)
    for (sid, uid), v in var.items():
        val = float(pulp.value(v) or 0.0)
        if val > 1e-9:
            allocations[uid].append((sid, val))
            sc = score_map.get((sid, uid), 0)
            total_pref += val * sc
            total_assigned += val
    res = {"status": status, "total_assigned": float(total_assigned), "total_pref_score": float(total_pref), "allocations": allocations}
    return res

if __name__ == "__main__":
    # Demo CLI for single instance
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--inst", required=True, help="instance json path")
    parser.add_argument("--out", default=None, help="output json path")
    args = parser.parse_args()

    with open(args.inst, "r", encoding="utf-8") as f:
        inst = json.load(f)

    res = run_lp_on_instance(inst)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            json.dump(res, fo, indent=2, ensure_ascii=False)
    else:
        import pprint

        pprint.pprint(res)