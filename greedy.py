#!/usr/bin/env python3
"""
Greedy baseline:
For each user (ordered by user id), assign from available suppliers in user's supplier_scores order (descending score)
until user's need is satisfied or suppliers exhausted.
"""
import json

def run_greedy_on_instance(inst):
    suppliers = {s["id"]: {"stock": int(s["stock"])} for s in inst["suppliers"]}
    users = inst["users"]
    allocations = {u["id"]: [] for u in users}
    total_pref = 0
    total_assigned = 0
    # order users by id (could use other policies)
    for user in users:
        need = int(user["need"])
        # sort supplier_scores by score desc
        prefs = sorted(user.get("supplier_scores", []), key=lambda x: -int(x[1]))
        for sid, score in prefs:
            if need <= 0:
                break
            if sid not in suppliers: 
                continue
            avail = suppliers[sid]["stock"]
            if avail <= 0:
                continue
            take = min(avail, need)
            suppliers[sid]["stock"] -= take
            need -= take
            allocations[user["id"]].append((sid, take))
            total_pref += take * int(score)
            total_assigned += take
    result = {"total_assigned": int(total_assigned), "total_pref_score": int(total_pref), "allocations": allocations}
    return result

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--inst", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    with open(args.inst, "r", encoding="utf-8") as f:
        inst = json.load(f)
    res = run_greedy_on_instance(inst)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    else:
        print(json.dumps(res, indent=2))
