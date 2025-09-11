#!/usr/bin/env python3
import argparse, glob, json, os, time
from mcmf import run_mcmf_on_instance
from greedy import run_greedy_on_instance
import traceback

try:
    from lp_baseline import run_lp_on_instance

    HAS_LP = True
except Exception:
    HAS_LP = False
try:
    from mcmf_warmstart import run_mcmf_with_warmstart

    HAS_WARM = True
except Exception:
    HAS_WARM = False
try:
    from mcmf_flow_scaling import run_mcmf_flow_scaling

    HAS_SCALING = True
except Exception:
    HAS_SCALING = False


def run_instance(path, args):
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
    warm_mcmf_res = None
    t_warm_mcmf = None
    scaling_mcmf_res = None
    t_scaling_mcmf = None
    if HAS_LP:
        try:
            t_lp0 = time.time()
            lp_res = run_lp_on_instance(inst)
            t_lp1 = time.time()
            lp_time = t_lp1 - t_lp0
        except Exception as e:
            lp_res = {"error": str(e)}
            lp_time = None
    if HAS_WARM and args.warm:
        try:
            topk = args.topk
            use_warmstart = args.warm
            t_warm_mcmf0 = time.time()
            warm_mcmf_res = run_mcmf_with_warmstart(inst, top_k=topk, use_warmstart=use_warmstart, verbose=True)
            t_warm_mcmf1 = time.time()
            t_warm_mcmf = t_warm_mcmf1 - t_warm_mcmf0
        except Exception as e:
            warm_mcmf_res = {"error": str(e), "traceback": traceback.format_exc()}
            t_warm_mcmf = None
    if HAS_SCALING:
        try:
            topk = args.topk
            t_scaling_mcmf0 = time.time()
            scaling_mcmf_res = run_mcmf_flow_scaling(inst, top_k=topk, verbose=False)
            t_scaling_mcmf1 = time.time()
            t_scaling_mcmf = t_scaling_mcmf1 - t_scaling_mcmf0
        except Exception as e:
            scaling_mcmf_res = {"error": str(e), "traceback": traceback.format_exc()}
            t_scaling_mcmf = None

    out = {"instance": name, "meta": inst.get("meta", {}),
           "mcmf": {"result": mres, "time": t1 - t0},
           "greedy": {"result": gres, "time": t2 - t1},
           "lp": {"result": lp_res, "time": lp_time},
           "warm_mcmf": {"result": warm_mcmf_res, "time": t_warm_mcmf},
           "scaling_mcmf": {"result": scaling_mcmf_res, "time": t_scaling_mcmf}}
    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, f"{name}_results.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote", outpath)
    return outpath


def main():
    import fnmatch
    p = argparse.ArgumentParser(description="Run solvers on instances (file, glob or directory).")
    p.add_argument("instances", nargs="+",
                   help="One or more instance files, globs (e.g. 'instances/*.json') or directories.")
    p.add_argument("--outdir", default="results")
    p.add_argument("--topk", type=int, default=None,
                   help="If set, run MCMF on per-user top-K suppliers (pruning).")
    p.add_argument("--warm", action="store_true",
                   help="Disable greedy warm-start (default: warm-start enabled when available).")
    p.add_argument("--recursive", action="store_true",
                   help="If a directory is provided, scan it recursively for matching files.")
    p.add_argument("--pattern", default="*.json",
                   help="Filename pattern to match when a directory is provided (default: '*.json').")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel worker processes to run (default 1 = sequential).")
    args = p.parse_args()

    # Expand inputs (files, globs, directories) -> list of file paths
    paths = []
    for entry in args.instances:
        # If looks like a glob pattern, try glob first
        if any(ch in entry for ch in "*?[]"):
            import glob
            matches = sorted(glob.glob(entry))
            if matches:
                paths.extend(matches)
            else:
                print(f"Warning: glob pattern '{entry}' did not match any files.")
            continue

        # If entry is an existing directory
        if os.path.isdir(entry):
            if args.recursive:
                for root, _, files in os.walk(entry):
                    for fname in files:
                        if fnmatch.fnmatch(fname, args.pattern):
                            paths.append(os.path.join(root, fname))
            else:
                import glob
                pattern = os.path.join(entry, args.pattern)
                matches = sorted(glob.glob(pattern))
                paths.extend(matches)
            continue

        # If entry is a file
        if os.path.isfile(entry):
            paths.append(entry)
            continue

        # fallback: try glob expansion anyway
        import glob
        matches = sorted(glob.glob(entry))
        if matches:
            paths.extend(matches)
        else:
            print(f"Warning: '{entry}' not found as file/dir/glob. Skipping.")

    # Deduplicate and sort
    paths = sorted(dict.fromkeys(paths))  # preserve order, unique
    if not paths:
        print("No instance files found. Exiting.")
        return

    # Prepare run dir
    os.makedirs(args.outdir, exist_ok=True)

    # If worker >1 use multiprocessing pool
    if args.workers and args.workers > 1:
        from multiprocessing import Pool
        # run_instance should accept (inst_path, args)
        tasks = [(p, args) for p in paths]
        with Pool(processes=args.workers) as pool:
            pool.starmap(run_instance, tasks)
    else:
        # sequential
        for inst_path in paths:
            run_instance(inst_path, args)


if __name__ == '__main__':
    main()
