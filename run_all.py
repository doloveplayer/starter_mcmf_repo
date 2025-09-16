#!/usr/bin/env python3
import argparse, json, os, time
from mcmf import run_mcmf_on_instance
from greedy import run_greedy_on_instance

try:
    from lp_baseline import run_lp_on_instance

    HAS_LP = True
except Exception:
    HAS_LP = False
try:
    from mcmf_greedy_warm import run_mcmf_with_warmstart

    HAS_WARM = True
except Exception:
    HAS_WARM = False
try:
    from mcmf_flow_scaling import run_mcmf_flow_scaling

    HAS_SCALING = True
except Exception:
    HAS_SCALING = False
try:
    from hungarian import hungarian_allocation

    HAS_HUNGARIAN = True
except Exception:
    HAS_HUNGARIAN = False

def run_instance(path, args):
    """
    对单个实例运行所有可用算法，并测量时间与内存峰值，输出 JSON。
    使用 measure_call 对每个算法统一测量（time, peak_rss_mb, tracemalloc_peak_mb）。
    """
    import threading, traceback, functools
    # 延迟导入 psutil/tracemalloc（保持兼容）
    try:
        import psutil
        HAS_PSUTIL = True
        _proc = psutil.Process(os.getpid())
    except Exception:
        HAS_PSUTIL = False
        _proc = None

    try:
        import tracemalloc
        HAS_TRACEMALLOC = True
    except Exception:
        HAS_TRACEMALLOC = False

    # 可调参数：采样间隔（秒），tracemalloc 是否默认启用
    SAMPLE_INTERVAL = 0.02  # 20 ms 采样间隔：精度/开销折中，可根据需要调小到 0.01
    USE_TRACEMALLOC = True  # 若你要禁用 tracemalloc 测量可改为 False

    def measure_call(func, *fargs, **fkwargs):
        """
        通用测量器：执行 func(*fargs, **fkwargs)，并返回字典：
          {"result","time","peak_rss_mb","py_tracemalloc_peak_mb","error"}
        - 使用 psutil + 线程定期采样 RSS 以近似峰值；
        - 使用 tracemalloc 捕获 Python 层的 peak（若可用且启用）。
        """
        res = {"result": None, "time": None, "peak_rss_mb": None, "py_tracemalloc_peak_mb": None, "error": None}

        stop_event = threading.Event()
        sampler_info = {"max_rss": 0}

        def sampler_loop(interval=SAMPLE_INTERVAL):
            # 在子线程中定期采样当前进程 RSS
            try:
                if HAS_PSUTIL and _proc is not None:
                    try:
                        rss0 = _proc.memory_info().rss
                        sampler_info["max_rss"] = max(sampler_info["max_rss"], rss0)
                    except Exception:
                        pass
                while not stop_event.is_set():
                    if HAS_PSUTIL and _proc is not None:
                        try:
                            rss = _proc.memory_info().rss
                            if rss > sampler_info["max_rss"]:
                                sampler_info["max_rss"] = rss
                        except Exception:
                            pass
                    time.sleep(interval)
                # final sample
                if HAS_PSUTIL and _proc is not None:
                    try:
                        rssf = _proc.memory_info().rss
                        sampler_info["max_rss"] = max(sampler_info["max_rss"], rssf)
                    except Exception:
                        pass
            except Exception:
                # ensure sampler thread never kills main flow
                return

        # 启用 tracemalloc（按 USE_TRACEMALLOC 且系统支持）
        if USE_TRACEMALLOC and HAS_TRACEMALLOC:
            try:
                tracemalloc.start()
            except Exception:
                pass

        sampler_thread = None
        if HAS_PSUTIL:
            sampler_thread = threading.Thread(target=sampler_loop, daemon=True)
            sampler_thread.start()

        # 执行 func（兼容关键字/位置调用）
        t0 = time.time()
        try:
            # 先试关键字调用（多数实现接受），若 TypeError 则退回位置调用
            try:
                result = func(*fargs, **fkwargs)
            except TypeError:
                result = func(*fargs)
            t1 = time.time()
            res["result"] = result
            res["time"] = t1 - t0
        except Exception as e:
            t1 = time.time()
            res["result"] = None
            res["time"] = t1 - t0
            res["error"] = f"{str(e)}\n{traceback.format_exc()}"

        # 停止 sampler_thread
        try:
            stop_event.set()
            if sampler_thread is not None:
                sampler_thread.join(timeout=1.0)
        except Exception:
            pass

        # 填写 peak RSS（MB）
        if HAS_PSUTIL:
            try:
                peak_rss = sampler_info.get("max_rss", 0)
                res["peak_rss_mb"] = round(float(peak_rss) / (1024.0 * 1024.0), 6)
            except Exception:
                res["peak_rss_mb"] = None
        else:
            res["peak_rss_mb"] = None

        # tracemalloc peak
        if USE_TRACEMALLOC and HAS_TRACEMALLOC:
            try:
                current, peak = tracemalloc.get_traced_memory()
                res["py_tracemalloc_peak_mb"] = round(float(peak) / (1024.0 * 1024.0), 6)
            except Exception:
                res["py_tracemalloc_peak_mb"] = None
            try:
                tracemalloc.stop()
            except Exception:
                pass
        else:
            res["py_tracemalloc_peak_mb"] = None

        return res

    # ------------- 主流程 -------------
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8") as f:
        inst = json.load(f)

    # mcmf
    mres_wrapper = measure_call(run_mcmf_on_instance, inst)

    # greedy
    gres_wrapper = measure_call(run_greedy_on_instance, inst)

    # lp 可选
    if HAS_LP:
        try:
            lp_wrapper = measure_call(run_lp_on_instance, inst)
        except Exception as e:
            lp_wrapper = {"result": {"error": str(e)}, "time": None, "peak_rss_mb": None, "py_tracemalloc_peak_mb": None,
                          "error": traceback.format_exc()}
    else:
        lp_wrapper = {"result": None, "time": None, "peak_rss_mb": None, "py_tracemalloc_peak_mb": None, "error": None}

    # warm mcmf 可选（注意签名兼容）
    warm_wrapper = {"result": None, "time": None, "peak_rss_mb": None, "py_tracemalloc_peak_mb": None, "error": None}
    if HAS_WARM and args.warm:
        try:
            # 如果 run_mcmf_with_warmstart 接受命名参数则该调用会成功，否则 measure_call 内部会回退到位置调用
            warm_wrapper = measure_call(run_mcmf_with_warmstart, inst, top_k=args.topk, use_warmstart=args.warm,
                                        verbose=True)
        except Exception as e:
            warm_wrapper = {"result": {"error": str(e)}, "time": None, "peak_rss_mb": None,
                            "py_tracemalloc_peak_mb": None, "error": traceback.format_exc()}

    # hungarian 可选
    hungarian_wrapper = {"result": None, "time": None, "peak_rss_mb": None, "py_tracemalloc_peak_mb": None, "error": None}
    if HAS_HUNGARIAN and args.hungarian:
        try:
            hungarian_wrapper = measure_call(hungarian_allocation, inst, max_expand=5000, verbose=True)
        except Exception as e:
            hungarian_wrapper = {"result": {"error": str(e)}, "time": None, "peak_rss_mb": None, "py_tracemalloc_peak_mb": None,
                                 "error": traceback.format_exc()}

    # 组织输出
    out = {
        "instance": name,
        "meta": inst.get("meta", {}),
        "mcmf": mres_wrapper,
        "greedy": gres_wrapper,
        "lp": lp_wrapper,
        "warm_mcmf": warm_wrapper,
        "hungarian": hungarian_wrapper,
        "run_timestamp": time.time()
    }

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, f"{name}_results.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
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
                   help="Enable greedy warm-start")
    p.add_argument("--hungarian", action="store_true",
                   help="Enable hungarian")
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
