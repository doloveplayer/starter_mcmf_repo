#!/usr/bin/env python3
# File: mcmf_warmstart.py
import time
import copy
from collections import defaultdict

from mcmf import MinCostMaxFlow
from greedy import run_greedy_on_instance

INF = 10 ** 18


def prune_instance_topk(inst, k):
    """
    Return a deep-copied instance where for each user we keep only top-k suppliers by score.
    k: int or None (if None, do nothing).
    """
    if k is None:
        return copy.deepcopy(inst)
    new_inst = {"suppliers": copy.deepcopy(inst.get("suppliers", [])),
                "users": [], "meta": copy.deepcopy(inst.get("meta", {}))}
    for user in inst.get("users", []):
        prefs = user.get("supplier_scores", [])
        # sort by score desc
        prefs_sorted = sorted(prefs, key=lambda x: -int(x[1]))
        kept = prefs_sorted[:k]
        new_user = {"id": user["id"], "need": user["need"], "supplier_scores": kept}
        new_inst["users"].append(new_user)
    # update meta (note: we keep original meta too)
    new_inst["meta"]["top_k"] = k
    return new_inst


def run_mcmf_with_warmstart(inst, top_k=None, use_warmstart=True, verbose=False):
    """
    Build graph similarly to original run_mcmf_on_instance, but:
      - optionally prune each user's supplier list to top_k
      - run greedy baseline to get warm-start allocation (if use_warmstart)
      - apply greedy allocation to residual graph (decrease forward caps, increase reverse caps)
      - run potentials init + SSAP solve
    Returns dict: {
        "total_flow": int,
        "total_pref_score": int,
        "allocations": {...},
        "timings": {"greedy_time": , "mcmf_time": , "total_time": }
    }
    """
    inst_proc = prune_instance_topk(inst, top_k) if top_k is not None else copy.deepcopy(inst)

    suppliers = inst_proc["suppliers"]
    users = inst_proc["users"]
    S = len(suppliers)
    U = len(users)
    total_nodes = 1 + S + U + 1
    s = 0
    t = total_nodes - 1

    # build MCMF graph (like original) but record edge indices for s->sup, sup->user, user->t
    mcmf = MinCostMaxFlow(total_nodes)
    supplier_id_to_index = {}
    s_to_sup_edge_idx = {}  # sup_node -> edge index in graph[s]
    for idx, sup in enumerate(suppliers):
        node_id = 1 + idx
        supplier_id_to_index[sup["id"]] = node_id
        # record index before adding
        idx1 = len(mcmf.graph[s])
        mcmf.add_edge(s, node_id, int(sup["stock"]), 0)
        s_to_sup_edge_idx[node_id] = idx1

    user_id_to_index = {}
    user_to_t_edge_idx = {}  # user_node -> idx in graph[user_node]
    for idx, user in enumerate(users):
        node_id = 1 + S + idx
        user_id_to_index[user["id"]] = node_id
        idx1 = len(mcmf.graph[node_id])
        mcmf.add_edge(node_id, t, int(user["need"]), 0)
        user_to_t_edge_idx[node_id] = idx1

    # add sup->user edges and record indices
    sup_user_edge_idx = {}  # (sup_node, user_node) -> idx in graph[sup_node]
    for user in users:
        u_node = user_id_to_index[user["id"]]
        for sid, score in user.get("supplier_scores", []):
            if sid in supplier_id_to_index:
                s_node = supplier_id_to_index[sid]
                idx1 = len(mcmf.graph[s_node])
                # capacity large (allow splitting) cost = -score
                mcmf.add_edge(s_node, u_node, 10 ** 9, -int(score))
                sup_user_edge_idx[(s_node, u_node)] = idx1

    # Optionally run greedy to obtain warm-start allocations
    greedy_time = 0.0
    greedy_alloc = None
    greedy_res = None
    if use_warmstart:
        t0 = time.time()
        greedy_res = run_greedy_on_instance(inst_proc)  # uses a copy internally
        greedy_time = time.time() - t0
        greedy_alloc = greedy_res.get("allocations", {})
        if verbose:
            print(
                f"[warmstart] greedy assigned total_pref={greedy_res.get('total_pref_score')} total_assigned={greedy_res.get('total_assigned')} in {greedy_time:.3f}s")

        # apply greedy allocation to residual graph:
        # For each user u: for each (sup_id, amt) allocated:
        #   - decrease forward cap on s->sup by amt, increase rev cap
        #   - decrease forward cap on sup->user by amt, increase rev cap
        #   - decrease forward cap on user->t by amt, increase rev cap
        # Note: we need to be careful not to create negative capacities (greedy should respect stock).
        for uid, lst in greedy_alloc.items():
            u_node = user_id_to_index.get(uid)
            if u_node is None:
                continue
            for sid, amt in lst:
                # get supplier node
                s_node = supplier_id_to_index.get(sid)
                if s_node is None:
                    continue
                # adjust s -> supplier edge (s to s_node)
                try:
                    eidx = s_to_sup_edge_idx[s_node]
                    fwd = mcmf.graph[s][eidx]
                    # find reverse edge index into supplier node
                    rev_idx = fwd[3]
                    # clamp
                    use_amt = min(amt, fwd[1])
                    if use_amt <= 0:
                        continue
                    fwd[1] -= use_amt
                    mcmf.graph[s_node][rev_idx][1] += use_amt
                except Exception:
                    # fallback: attempt to find correct edge by scan
                    for i, e in enumerate(mcmf.graph[s]):
                        if e[0] == s_node:
                            use_amt = min(amt, e[1])
                            e[1] -= use_amt
                            mcmf.graph[s_node][e[3]][1] += use_amt
                            break

                # adjust supplier -> user edge
                key = (s_node, u_node)
                if key in sup_user_edge_idx:
                    eidx = sup_user_edge_idx[key]
                    fwd = mcmf.graph[s_node][eidx]
                    rev_idx = fwd[3]
                    use_amt = min(amt, fwd[1])
                    if use_amt <= 0:
                        continue
                    fwd[1] -= use_amt
                    mcmf.graph[u_node][rev_idx][1] += use_amt
                else:
                    # the edge might not exist after pruning; skip
                    continue

                # adjust user -> t edge
                try:
                    eidx = user_to_t_edge_idx[u_node]
                    fwd = mcmf.graph[u_node][eidx]
                    rev_idx = fwd[3]
                    use_amt = min(amt, fwd[1])
                    if use_amt <= 0:
                        continue
                    fwd[1] -= use_amt
                    mcmf.graph[t][rev_idx][1] += use_amt
                except Exception:
                    for i, e in enumerate(mcmf.graph[u_node]):
                        if e[0] == t:
                            use_amt = min(amt, e[1])
                            e[1] -= use_amt
                            mcmf.graph[t][e[3]][1] += use_amt
                            break

    # Now run MCMF solve (init potentials then SSAP)
    t0 = time.time()
    # initialize potentials using Bellman-Ford (this method uses current residual costs)
    mcmf.init_potential(s)
    mcmf_total_flow, mcmf_total_cost = mcmf.solve(s, t)
    mcmf_time = time.time() - t0

    # --- rebuild full allocations & compute totals from residual graph (robust) ---
    allocations = {u["id"]: [] for u in users}
    total_flow_all = 0.0
    total_pref_all = 0.0

    # build a score lookup map (sup_node,user_node) -> score
    score_map = {}
    for user in users:
        u_node = user_id_to_index[user["id"]]
        for sid, score in user.get("supplier_scores", []):
            if sid in supplier_id_to_index:
                s_node = supplier_id_to_index[sid]
                score_map[(s_node, u_node)] = int(score)

    for sup in suppliers:
        sup_node = supplier_id_to_index[sup["id"]]
        for idx, (v, cap, cost, rev) in enumerate(mcmf.graph[sup_node]):
            if 1 + S <= v <= S + U:
                user_node = v
                # the reverse edge at user_node index 'rev' stores the flow assigned on forward edge:
                allocated = mcmf.graph[user_node][rev][1]
                if allocated > 0:
                    user_id = users[user_node - 1 - S]["id"]
                    allocations[user_id].append((sup["id"], int(allocated)))
                    total_flow_all += allocated
                    sc = score_map.get((sup_node, user_node), 0)
                    total_pref_all += allocated * sc

    total_flow_all = int(total_flow_all)
    total_pref_all = int(total_pref_all)
    result = {
        "total_flow": total_flow_all,
        "total_pref_score": total_pref_all,
        "allocations": allocations,
        "timings": {"greedy_time": greedy_time, "mcmf_time": mcmf_time, "total_time": greedy_time + mcmf_time}
    }
    return result


# If module run as script, a tiny demo
if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser()
    p.add_argument("--inst", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--no-warm", action="store_true", help="disable greedy warm-start")
    args = p.parse_args()
    with open(args.inst, "r", encoding="utf-8") as f:
        inst = json.load(f)
    res = run_mcmf_with_warmstart(inst, top_k=args.topk, use_warmstart=not args.no_warm, verbose=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    else:
        print(json.dumps(res, indent=2))
