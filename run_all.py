#!/usr/bin/env python3
import argparse, glob, json, os, time
from mcmf import run_mcmf_on_instance
from greedy import run_greedy_on_instance
try:
    from lp_baseline import run_lp_on_instance
    HAS_LP = True
except Exception:
    HAS_LP = False


def run_instance(path, outdir):
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8") as f:
        inst = json.load(f)
    t0 = time.time()
    mres = run_mcmf_on_instance(inst)
    t1 = time.time()
    gres = run_greedy_on_instance(inst)
    t2 = time.time()
    lp_res = None
    lp_time = None
    if HAS_LP:
        try:
            t_lp0 = time.time()
            lp_res = run_lp_on_instance(inst)
            t_lp1 = time.time()
            lp_time = t_lp1 - t_lp0
        except Exception as e:
            lp_res = {"error": str(e)}
            lp_time = None
    out = {"instance": name, "meta": inst.get("meta", {}),
           "mcmf": {"result": mres, "time": t1-t0},
           "greedy": {"result": gres, "time": t2-t1},
           "lp": {"result": lp_res, "time": lp_time}}
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{name}_results.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote", outpath)
    return outpath

def main():
    p = argparse.ArgumentParser()
    p.add_argument("instances", nargs="+")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    for inst in args.instances:
        run_instance(inst, args.outdir)

if __name__ == '__main__':
    main()
