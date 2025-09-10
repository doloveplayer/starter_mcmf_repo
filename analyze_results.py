#!/usr/bin/env python3
"""
analyze_results.py

功能：
- 读取多个 results/*.json（由 run_all.py 产生），汇总为 CSV；
- 生成对比图表（总偏好、运行时间、满足率等）并保存为 PNG；
- 可选：通过 --instances-dir 读取原始实例以计算总需求 (total_demand)；
- 可选：如果安装了 scipy，会对 mcmf vs greedy / lp 做配对 t 检验。

用法示例：
python analyze_results.py results/*.json --out summary.csv --plotdir plots --instances-dir instances

"""
import argparse
import glob
import json
import math
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

# optional
try:
    from scipy import stats

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def load_instance_total_demand(instances_dir, instance_name):
    """
    尝试从 instances_dir 中加载名为 instance_name.json 的实例并返回 total_demand（sum of user needs）
    若不存在或读取失败，则返回 None
    """
    p = Path(instances_dir) / f"{instance_name}.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            inst = json.load(f)
        total = 0
        for u in inst.get("users", []):
            total += int(u.get("need", 0))
        return total
    except Exception:
        return None


def parse_result_file(path, instances_dir=None):
    """
    解析单个 result JSON 文件，返回一字典包含我们关心的字段（可能为 None）
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    name = data.get("instance", Path(path).stem)
    meta = data.get("meta", {})

    # helper to safely extract nested values
    def get_nested(d, *keys, default=None):
        cur = d
        for k in keys:
            if cur is None:
                return default
            cur = cur.get(k, None)
        return cur if cur is not None else default

    # MCMF
    mcmf_result = get_nested(data, "mcmf", "result", default=None)
    mcmf_time = get_nested(data, "mcmf", "time", default=None)
    mcmf_total_pref = None
    mcmf_total_flow = None
    if isinstance(mcmf_result, dict):
        mcmf_total_pref = mcmf_result.get("total_pref_score", None)
        mcmf_total_flow = mcmf_result.get("total_flow", None)

    # Greedy
    greedy_result = get_nested(data, "greedy", "result", default=None)
    greedy_time = get_nested(data, "greedy", "time", default=None)
    greedy_total_pref = None
    greedy_total_assigned = None
    if isinstance(greedy_result, dict):
        greedy_total_pref = greedy_result.get("total_pref_score", None)
        greedy_total_assigned = greedy_result.get("total_assigned", None)

    # LP baseline (may be None if pulp not installed)
    lp_result = get_nested(data, "lp", "result", default=None)
    lp_time = get_nested(data, "lp", "time", default=None)
    lp_total_pref = None
    lp_total_assigned = None
    if isinstance(lp_result, dict):
        lp_total_pref = lp_result.get("total_pref_score", None)
        # different LP implementation used 'total_assigned' naming
        lp_total_assigned = lp_result.get("total_assigned", None)

    # Warm MCMF
    warm_mcmf_result = get_nested(data, "warm_mcmf", "result", default=None)
    warm_time = get_nested(data, "warm_mcmf", "time", default=None)
    warm_total_pref = None
    warm_total_assigned = None
    warm_timings = None
    if isinstance(warm_mcmf_result, dict):
        warm_total_pref = warm_mcmf_result.get("total_pref_score", None)
        warm_total_assigned = warm_mcmf_result.get("total_flow", None)
        warm_timings = warm_mcmf_result.get("timings", None)

    # optionally compute total_demand by loading instance file
    total_demand = None
    if instances_dir is not None:
        total_demand = load_instance_total_demand(instances_dir, name)

    return {
        "instance": name,
        "S": meta.get("S"),
        "U": meta.get("U"),
        "avg_degree": meta.get("avg_degree"),
        "mcmf_total_pref": try_float(mcmf_total_pref),
        "mcmf_total_flow": try_float(mcmf_total_flow),
        "mcmf_time": try_float(mcmf_time),
        "greedy_total_pref": try_float(greedy_total_pref),
        "greedy_total_assigned": try_float(greedy_total_assigned),
        "greedy_time": try_float(greedy_time),
        "lp_total_pref": try_float(lp_total_pref),
        "lp_total_assigned": try_float(lp_total_assigned),
        "lp_time": try_float(lp_time),
        "warm_mcmf_total_pref": try_float(warm_total_pref),
        "warm_mcmf_total_assigned": try_float(warm_total_assigned),
        "warm_mcmf_time": try_float(warm_timings.total_time if warm_timings is not None else None),
        "warm_mcmf_greed_time": try_float(warm_timings.greedy_time if warm_timings is not None else None),
        "warm_mcmf_mcmf_time": try_float(warm_timings.mcmf_time if warm_timings is not None else None),
        "total_demand": try_float(total_demand)
    }


def try_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def aggregate_results(rows):
    df = pd.DataFrame(rows)
    # computed columns
    # fulfillment: assigned / total_demand if available
    if "total_demand" in df.columns:
        df["mcmf_fulfillment"] = df.apply(lambda r: safe_divide(r.get("mcmf_total_flow"), r.get("total_demand")),
                                          axis=1)
        df["greedy_fulfillment"] = df.apply(
            lambda r: safe_divide(r.get("greedy_total_assigned"), r.get("total_demand")), axis=1)
        df["lp_fulfillment"] = df.apply(lambda r: safe_divide(r.get("lp_total_assigned"), r.get("total_demand")),
                                        axis=1)
        df["warm_mcmf_fulfillment"] = df.apply(
            lambda r: safe_divide(r.get("warm_mcmf_total_assigned"), r.get("total_demand")), axis=1)
    else:
        df["mcmf_fulfillment"] = None
        df["greedy_fulfillment"] = None
        df["lp_fulfillment"] = None
        df["warm_mcmf_fulfillment"] = None

    # improvement percent over greedy and lp
    df["impr_over_greedy_pct"] = df.apply(
        lambda r: percent_improvement(r.get("mcmf_total_pref"), r.get("greedy_total_pref")), axis=1)
    df["dis_between_lp_pct"] = df.apply(lambda r: percent_improvement(r.get("lp_total_pref"), r.get("mcmf_total_pref")),
                                        axis=1)
    df["dis_between_warm_pct"] = df.apply(
        lambda r: percent_improvement(r.get("warm_mcmf_total_pref"), r.get("mcmf_total_pref")), axis=1)

    return df


def safe_divide(a, b):
    try:
        if a is None or b is None:
            return None
        b_f = float(b)
        if b_f == 0:
            return None
        return float(a) / b_f
    except Exception:
        return None


def percent_improvement(a, b):
    # (a - b) / max(|b|, eps)
    if a is None or b is None:
        return None
    denom = max(abs(b), 1e-9)
    return 100.0 * (a - b) / denom


def save_csv(df, out_path):
    df.to_csv(out_path, index=False)
    print(f"Wrote CSV summary to {out_path}")


def plot_comparisons(df, plotdir, max_instances_for_bar=20):
    os.makedirs(plotdir, exist_ok=True)
    # determine methods available
    methods = ["mcmf", "greedy", "lp", "warm_mcmf"]
    # Only include methods with at least one non-null total_pref
    available = []
    for m in methods:
        col = f"{m}_total_pref"
        if col in df.columns and df[col].notnull().any():
            available.append(m)

    # If many instances, plot aggregated statistics instead of wide grouped bar
    ninst = len(df)
    if ninst == 0:
        print("No instances to plot.")
        return

    # 1) Total preference comparison: grouped bar per instance (or mean+std if many)
    pref_cols = [f"{m}_total_pref" for m in available]
    if ninst <= max_instances_for_bar:
        ax = df[pref_cols].plot.bar(figsize=(max(8, ninst * 0.6), 6))
        ax.set_title("Total preference (higher is better) - per instance")
        ax.set_xlabel("Instance (index)")
        ax.set_ylabel("Total preference")
        plt.tight_layout()
        fpath = os.path.join(plotdir, "total_pref_per_instance.png")
        plt.savefig(fpath)
        plt.close()
        print("Saved", fpath)
    else:
        # aggregated mean+std
        stats_df = df[pref_cols].agg(["mean", "std"]).transpose().reset_index().rename(columns={"index": "method"})
        stats_df["method"] = stats_df["method"].str.replace("_total_pref", "")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(stats_df["method"], stats_df["mean"], yerr=stats_df["std"], capsize=5)
        ax.set_title("Mean total preference ± std (across instances)")
        ax.set_ylabel("Total preference")
        plt.tight_layout()
        fpath = os.path.join(plotdir, "total_pref_mean.png")
        plt.savefig(fpath)
        plt.close()
        print("Saved", fpath)

    # 2) Runtime comparison
    time_cols = [f"{m}_time" for m in available]
    if df[time_cols].notnull().any().any():
        if ninst <= max_instances_for_bar:
            ax = df[time_cols].plot.bar(figsize=(max(8, ninst * 0.6), 6))
            ax.set_title("Runtime (seconds) - per instance")
            ax.set_xlabel("Instance (index)")
            ax.set_ylabel("Time (s)")
            plt.tight_layout()
            fpath = os.path.join(plotdir, "runtime_per_instance.png")
            plt.savefig(fpath)
            plt.close()
            print("Saved", fpath)
        else:
            stats_df = df[time_cols].agg(["mean", "std"]).transpose().reset_index().rename(columns={"index": "method"})
            stats_df["method"] = stats_df["method"].str.replace("_time", "")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(stats_df["method"], stats_df["mean"], yerr=stats_df["std"], capsize=5)
            ax.set_title("Mean runtime ± std (across instances)")
            ax.set_ylabel("Time (s)")
            plt.tight_layout()
            fpath = os.path.join(plotdir, "runtime_mean.png")
            plt.savefig(fpath)
            plt.close()
            print("Saved", fpath)

    # 3) Fulfillment rate if available
    if df["mcmf_fulfillment"].notnull().any():
        ful_cols = [c for c in ["mcmf_fulfillment", "greedy_fulfillment", "lp_fulfillment", "warm_mcmf_fulfillment"] if
                    c in df.columns and df[c].notnull().any()]
        if len(ful_cols) > 0:
            if ninst <= max_instances_for_bar:
                ax = df[ful_cols].plot.bar(figsize=(max(8, ninst * 0.6), 6))
                ax.set_title("Fulfillment rate (assigned / total demand) - per instance")
                ax.set_xlabel("Instance (index)")
                ax.set_ylabel("Fulfillment rate")
                plt.tight_layout()
                fpath = os.path.join(plotdir, "fulfillment_per_instance.png")
                plt.savefig(fpath)
                plt.close()
                print("Saved", fpath)
            else:
                stats_df = df[ful_cols].agg(["mean", "std"]).transpose().reset_index().rename(
                    columns={"index": "method"})
                stats_df["method"] = stats_df["method"].str.replace("_fulfillment", "")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(stats_df["method"], stats_df["mean"], yerr=stats_df["std"], capsize=5)
                ax.set_title("Mean fulfillment rate ± std (across instances)")
                ax.set_ylabel("Fulfillment rate")
                plt.tight_layout()
                fpath = os.path.join(plotdir, "fulfillment_mean.png")
                plt.savefig(fpath)
                plt.close()
                print("Saved", fpath)

    # 4) Scatter: runtime vs total_pref for each method (if time data exists)
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False
    colors = {"mcmf": "C0", "greedy": "C1", "lp": "C2", "warm_mcmf": "C3"}
    for m in available:
        x = df[f"{m}_time"]
        y = df[f"{m}_total_pref"]
        if x.notnull().any() and y.notnull().any():
            ax.scatter(x, y, label=m, color=colors.get(m, "C0"), alpha=0.7)
            plotted = True
    if plotted:
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("Total preference")
        ax.set_title("Runtime vs Total preference")
        ax.legend()
        plt.tight_layout()
        fpath = os.path.join(plotdir, "runtime_vs_pref.png")
        plt.savefig(fpath)
        plt.close()
        print("Saved", fpath)
    else:
        plt.close()

def statistical_tests(df):
    """
    可选的配对 t 检验（若 scipy 可用）：
      - MCMF vs Greedy on total_pref
      - MCMF vs LP on total_pref (若 LP 存在)
    返回字典结果
    """
    res = {}
    if not HAS_SCIPY:
        print("scipy not available: skipping statistical tests")
        return res
    # prepare paired arrays (drop NaNs)
    a = df["mcmf_total_pref"]
    b = df["greedy_total_pref"]
    mask = a.notnull() & b.notnull()
    if mask.any():
        t, p = stats.ttest_rel(a[mask], b[mask])
        res["mcmf_vs_greedy_t"] = float(t)
        res["mcmf_vs_greedy_p"] = float(p)
    # mcmf vs lp
    if "lp_total_pref" in df.columns:
        c = df["lp_total_pref"]
        mask2 = a.notnull() & c.notnull()
        if mask2.any():
            t2, p2 = stats.ttest_rel(a[mask2], c[mask2])
            res["mcmf_vs_lp_t"] = float(t2)
            res["mcmf_vs_lp_p"] = float(p2)
    if "warm_mcmf_total_pref" in df.columns:
        d = df["warm_mcmf_total_pref"]
        mask2 = a.notnull() & d.notnull()
        if mask2.any():
            t2, p2 = stats.ttest_rel(a[mask2], d[mask2])
            res["mcmf_vs_lp_t"] = float(t2)
            res["mcmf_vs_lp_p"] = float(p2)
    return res


def main():
    p = argparse.ArgumentParser(description="Aggregate JSON result files and produce CSV + plots")
    p.add_argument("results", nargs="+", help="one or more result JSON files or glob pattern (e.g. results/*.json)")
    p.add_argument("--out", default="summary.csv", help="CSV output file")
    p.add_argument("--plotdir", default="plots", help="directory to save plots")
    p.add_argument("--instances-dir", default=None,
                   help="optional directory containing instance JSONs to compute total demand (names must match instance field)")
    args = p.parse_args()

    # expand globs and directories
    paths = []
    for r in args.results:
        if "*" in r or "?" in r or "[" in r:
            paths.extend(glob.glob(r))
        elif os.path.isdir(r):
            # add all json files in dir
            paths.extend(glob.glob(os.path.join(r, "*.json")))
        else:
            paths.append(r)
    paths = sorted([p for p in paths if os.path.isfile(p)])
    if not paths:
        print("No result files found. Check your path/glob.")
        return

    rows = []
    for pth in paths:
        try:
            row = parse_result_file(pth, instances_dir=args.instances_dir)
            rows.append(row)
        except Exception as e:
            print(f"Warning: failed to parse {pth}: {e}")

    df = aggregate_results(rows)
    save_csv(df, args.out)

    # print brief summary to console
    print("\nSummary statistics (means across instances):")
    summary = df.mean(numeric_only=True).to_dict()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # optional statistical tests
    stats_res = statistical_tests(df)
    if stats_res:
        print("\nStatistical test results:")
        for k, v in stats_res.items():
            print(f"  {k}: {v}")

    # generate plots
    plot_comparisons(df, args.plotdir)


if __name__ == "__main__":
    main()
