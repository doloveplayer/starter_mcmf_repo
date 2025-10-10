#!/usr/bin/env python3
# greedy_float.py -- greedy baseline supporting float stocks/needs

import json

EPS = 1e-9

def run_greedy_on_instance(inst):
    """
    Greedy baseline that supports float stocks and needs.
    Allocates per-user in descending score order until user's need satisfied or suppliers exhausted.
    Returns totals as floats and allocations (amount floats).
    """
    # initialize suppliers with float stock
    suppliers = {s["id"]: {"stock": float(s.get("stock", 0.0))} for s in inst.get("suppliers", [])}
    users = inst.get("users", [])
    allocations = {u["id"]: [] for u in users}
    total_pref = 0.0
    total_assigned = 0.0
    # iterate users in given order (or could sort by urgency)
    for user in users:
        need = float(user.get("need", 0.0))
        # Optionally cap very small needs
        if need <= EPS:
            continue
        # sort preference list by score desc
        prefs = sorted(user.get("supplier_scores", []), key=lambda x: -float(x[1]))
        for sid, score in prefs:
            if need <= EPS:
                break
            if sid not in suppliers:
                continue
            avail = suppliers[sid]["stock"]
            if avail <= EPS:
                continue
            # take min as floats
            take = min(avail, need)
            # subtract
            suppliers[sid]["stock"] = suppliers[sid]["stock"] - take
            need = need - take
            allocations[user["id"]].append((sid, float(take)))
            total_pref += float(take) * float(score)
            total_assigned += float(take)
    result = {"total_assigned": float(total_assigned), "total_pref_score": float(total_pref), "allocations": allocations}
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--inst", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    with open(args.inst, "r", encoding="utf-8") as f:
        inst = json.load(f)
    res = run_greedy_on_instance(inst)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            json.dump(res, fo, indent=2)
    else:
        print(json.dumps(res, indent=2))
