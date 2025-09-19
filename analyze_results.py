#!/usr/bin/env python3
"""
analyze_results.py

功能：
- 读取多个 results/*.json（由 run_all.py 产生），汇总为 CSV；
- 生成对比图表（总偏好、运行时间、满足率、内存开销等）并保存为 PNG；
- 可选：通过 --instances-dir 读取原始实例以计算总需求 (total_demand)；
- 可选：如果安装了 scipy，会对 mcmf vs greedy / lp 做配对 t 检验。

用法示例：
python analyze_results.py results/*.json --out summary.csv --plotdir plots --instances-dir instances

"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# optional
try:
    from scipy import stats

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def load_instance_user_needs(instances_dir, instance_name):
    """
    从 instances_dir/instance_name.json 中加载 per-user needs，
    返回 dict {user_id: need}（need 为 int），若失败返回 None。
    """
    if instances_dir is None:
        return None
    p = Path(instances_dir) / f"{instance_name}.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            inst = json.load(f)
        needs = {}
        for u in inst.get("users", []):
            uid = u.get("id")
            need = u.get("need", None)
            if uid is not None and need is not None:
                try:
                    needs[str(uid)] = int(need)
                except Exception:
                    try:
                        needs[str(uid)] = int(float(need))
                    except Exception:
                        needs[str(uid)] = None
        return needs
    except Exception:
        return None


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


def extract_alloc_map(result_dict):
    """
    从 result_dict（例如 data['mcmf']['result']）中提取 user -> allocated_amount 映射。
    返回 (alloc_map: dict user_id->total_allocated_float, total_users_in_alloc: int)
    若无法解析则返回 (None, None)
    """
    if not isinstance(result_dict, dict):
        return None, None

    allocs = result_dict.get("allocations") or result_dict.get("allocation") or result_dict.get("alloc") or None
    if allocs is None:
        # 尝试从任何 dict 字段找到类似 allocations 的结构
        for k, v in result_dict.items():
            if isinstance(v, dict):
                # heuristics: dict of user_id -> list/tuple
                sample_vals = list(v.values())[:3]
                if sample_vals and all(isinstance(x, (list, tuple, dict)) for x in sample_vals):
                    allocs = v
                    break
        if allocs is None:
            return None, None

    # 标准化为 dict user_id -> list
    if isinstance(allocs, list):
        alloc_map = {}
        # 常见 list 元素类型： (user_id, [(sup,amt),...]) 或 dict {'user':id,'alloc':[...] }
        for item in allocs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                uid = item[0]
                lst = item[1]
                alloc_map[str(uid)] = lst
            elif isinstance(item, dict):
                uid = item.get("user") or item.get("uid") or item.get("id")
                lst = item.get("allocations") or item.get("alloc") or item.get("assigned") or item.get("allocation")
                if uid is not None and lst is not None:
                    alloc_map[str(uid)] = lst
        if not alloc_map:
            return None, None
    elif isinstance(allocs, dict):
        alloc_map = {str(k): v for k, v in allocs.items()}
    else:
        return None, None

    # 汇总每个 user 的总分配量（将 list -> 总和）
    out_map = {}
    for uid, lst in alloc_map.items():
        if not lst:
            out_map[str(uid)] = 0.0
            continue
        # 如果 lst 是数值（极少见），直接用
        if isinstance(lst, (int, float)):
            out_map[str(uid)] = float(lst)
            continue
        total = 0.0
        for entry in lst:
            amt = 0.0
            if entry is None:
                continue
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                # (sup, amt, ...)
                try:
                    amt = float(entry[1])
                except Exception:
                    try:
                        amt = float(str(entry[1]))
                    except Exception:
                        amt = 0.0
            elif isinstance(entry, dict):
                # try common keys
                found = False
                for key in ("amt", "amount", "assigned", "quantity", "flow"):
                    if key in entry:
                        try:
                            amt = float(entry[key])
                            found = True
                            break
                        except Exception:
                            amt = 0.0
                if not found:
                    # try to pick first numeric value
                    for v in entry.values():
                        if isinstance(v, (int, float)):
                            amt = float(v)
                            break
            else:
                try:
                    amt = float(entry)
                except Exception:
                    amt = 0.0
            total += amt
        out_map[str(uid)] = total

    return out_map, len(out_map)


def compute_satisfaction_stats(alloc_map, user_needs_map, total_users_from_meta=None):
    """
    给定 alloc_map: user_id->allocated_amount (float)
          user_needs_map: user_id->need (int)  或 None
    计算满足率 >=100%, >=80%, >=60% 的用户数量与占比。
    如果 user_needs_map 为 None 则返回 (None..)
    返回 dict 包含 counts 和 shares，若无法计算则值为 None。
    """
    if alloc_map is None:
        return {
            "full_count": None, "full_share": None,
            "p80_count": None, "p80_share": None,
            "p60_count": None, "p60_share": None,
            "total_users_in_alloc": None
        }
    # 如果没有 user_needs_map，无法计算准确比例
    if user_needs_map is None:
        return {
            "full_count": None, "full_share": None,
            "p80_count": None, "p80_share": None,
            "p60_count": None, "p60_share": None,
            "total_users_in_alloc": len(alloc_map)
        }

    full = 0
    p80 = 0
    p60 = 0
    total_users = 0
    for uid, alloc in alloc_map.items():
        total_users += 1
        need = user_needs_map.get(str(uid))
        if need is None or need == 0:
            # skip users with unknown or zero need
            continue
        ratio = float(alloc) / float(need)
        if ratio >= 1.0:
            full += 1
        if ratio >= 0.8:
            p80 += 1
        if ratio >= 0.6:
            p60 += 1

    # denom for share: prefer total_users_from_meta if provided and >0, else number of users with known need
    denom = None
    if total_users_from_meta is not None:
        try:
            if int(total_users_from_meta) > 0:
                denom = int(total_users_from_meta)
        except Exception:
            denom = None
    if denom is None:
        # count users with known need among alloc_map
        known_users = sum(1 for uid in alloc_map.keys() if user_needs_map.get(str(uid)) is not None)
        denom = known_users if known_users > 0 else None

    def share(count):
        if denom is None:
            return None
        return float(count) / float(denom)

    return {
        "full_count": int(full), "full_share": share(full),
        "p80_count": int(p80), "p80_share": share(p80),
        "p60_count": int(p60), "p60_share": share(p60),
        "total_users_in_alloc": len(alloc_map)
    }


def parse_result_file(path, instances_dir=None):
    """
    解析单个 result JSON 文件，返回一字典包含我们关心的字段（可能为 None）。
    现在额外统计每种方法中：未被分配用户数/占比，以及满足 >=100%, >=80%, >=60% 的用户数/占比。
    并将每个算法的运行时间与内存峰值作为单独的字段返回（mcmf_time, mcmf_peak_rss_mb, ...）。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    name = data.get("instance", Path(path).stem)
    meta = data.get("meta", {})

    # try load per-user needs from instances_dir (preferred)
    user_needs_map = load_instance_user_needs(instances_dir, name)

    # helper to safely extract nested values
    def get_nested(d, *keys, default=None):
        cur = d
        for k in keys:
            if cur is None:
                return default
            cur = cur.get(k, None)
        return cur if cur is not None else default

    # ---- helper to get counts using functions above ----
    def analyze_method(result_dict):
        # unserved count/share (reuse logic)
        alloc_map, total_users_in_alloc = extract_alloc_map(result_dict)
        # unserved: users with alloc == 0
        if alloc_map is None:
            unserved_count = None
            unserved_share = None
        else:
            unserved_count = sum(1 for v in alloc_map.values() if (v is None or float(v) <= 0.0))
            # denom for share: prefer meta['U'] if present
            denom = None
            try:
                denom = int(meta.get("U")) if meta.get("U") is not None else None
            except Exception:
                denom = None
            if denom is None:
                denom = total_users_in_alloc if total_users_in_alloc > 0 else None
            unserved_share = None if denom is None else float(unserved_count) / float(denom)

        # satisfaction levels (needs necessary)
        sat_stats = compute_satisfaction_stats(alloc_map, user_needs_map, total_users_from_meta=meta.get("U"))

        # total_pref / total_flow extraction (if present)
        total_pref = None
        total_flow = None
        if isinstance(result_dict, dict):
            total_pref = result_dict.get("total_pref_score") or result_dict.get("total_pref") or result_dict.get(
                "pref") or None
            # some results record 'total_assigned' or 'total_flow'
            total_flow = result_dict.get("total_flow") or result_dict.get("total_assigned") or result_dict.get(
                "assigned") or None

        # return aggregated stats
        return {
            "unserved_count": try_int(unserved_count),
            "unserved_share": try_float(unserved_share),
            "full_count": try_int(sat_stats["full_count"]),
            "full_share": try_float(sat_stats["full_share"]),
            "p80_count": try_int(sat_stats["p80_count"]),
            "p80_share": try_float(sat_stats["p80_share"]),
            "p60_count": try_int(sat_stats["p60_count"]),
            "p60_share": try_float(sat_stats["p60_share"]),
            "total_pref": try_float(total_pref),
            "total_flow": try_float(total_flow)
        }

    # parse each method's result AND time & memory (try common locations)
    mcmf_result = get_nested(data, "mcmf", "result", default=None)
    mcmf_time = get_nested(data, "mcmf", "time", default=None) or get_nested(data, "mcmf", "timings", "total_time",
                                                                             default=None)
    mcmf_peak_rss = get_nested(data, "mcmf", "peak_rss_mb", default=None)
    mcmf_tracemalloc = get_nested(data, "mcmf", "py_tracemalloc_peak_mb", default=None)
    m = analyze_method(mcmf_result)

    greedy_result = get_nested(data, "greedy", "result", default=None)
    greedy_time = get_nested(data, "greedy", "time", default=None) or get_nested(data, "greedy", "timings",
                                                                                 "total_time", default=None)
    greedy_peak_rss = get_nested(data, "greedy", "peak_rss_mb", default=None)
    greedy_tracemalloc = get_nested(data, "greedy", "py_tracemalloc_peak_mb", default=None)
    g = analyze_method(greedy_result)

    lp_result = get_nested(data, "lp", "result", default=None)
    lp_time = get_nested(data, "lp", "time", default=None) or get_nested(data, "lp", "timings", "total_time",
                                                                         default=None)
    lp_peak_rss = get_nested(data, "lp", "peak_rss_mb", default=None)
    lp_tracemalloc = get_nested(data, "lp", "py_tracemalloc_peak_mb", default=None)
    l = analyze_method(lp_result)

    warm_result = get_nested(data, "warm_mcmf", "result", default=None)
    warm_time = get_nested(data, "warm_mcmf", "time", default=None) or get_nested(data, "warm_mcmf", "timings",
                                                                                  "total_time", default=None)
    warm_peak_rss = get_nested(data, "warm_mcmf", "peak_rss_mb", default=None)
    warm_tracemalloc = get_nested(data, "warm_mcmf", "py_tracemalloc_peak_mb", default=None)
    w = analyze_method(warm_result)

    hungarian_result = get_nested(data, "hungarian", "result", default=None)
    hungarian_time = get_nested(data, "hungarian", "time", default=None) or get_nested(data, "hungarian", "timings",
                                                                                       "total_time", default=None)
    hungarian_peak_rss = get_nested(data, "hungarian", "peak_rss_mb", default=None)
    hungarian_tracemalloc = get_nested(data, "hungarian", "py_tracemalloc_peak_mb", default=None)
    h = analyze_method(hungarian_result)

    pso_result = get_nested(data, "pso", "result", default=None)
    pso_time = get_nested(data, "pso", "time", default=None) or get_nested(data, "hungarian", "timings", "total_time",
                                                                           default=None)
    pso_peak_rss = get_nested(data, "pso", "peak_rss_mb", default=None)
    pso_tracemalloc = get_nested(data, "pso", "py_tracemalloc_peak_mb", default=None)
    p = analyze_method(pso_result)

    # optionally compute total_demand by loading instance file (as before)
    total_demand = None
    if instances_dir is not None:
        total_demand = load_instance_total_demand(instances_dir, name)

    # now build returned row (keys chosen to be consistent / descriptive)
    row = {
        "instance": name,
        "S": meta.get("S"),
        "U": meta.get("U"),
        "avg_degree": meta.get("avg_degree"),
        # mcmf
        "mcmf_total_pref": m["total_pref"],
        "mcmf_total_flow": m["total_flow"],
        "mcmf_time": try_float(mcmf_time),
        "mcmf_peak_rss_mb": try_float(mcmf_peak_rss),
        "mcmf_py_tracemalloc_peak_mb": try_float(mcmf_tracemalloc),
        "mcmf_unserved_share": m["unserved_share"],
        "mcmf_full_share": m["full_share"],
        "mcmf_p80_share": m["p80_share"],
        "mcmf_p60_share": m["p60_share"],
        # greedy
        "greedy_total_pref": g["total_pref"],
        "greedy_total_assigned": g["total_flow"],
        "greedy_time": try_float(greedy_time),
        "greedy_peak_rss_mb": try_float(greedy_peak_rss),
        "greedy_py_tracemalloc_peak_mb": try_float(greedy_tracemalloc),
        "greedy_unserved_share": g["unserved_share"],
        "greedy_full_share": g["full_share"],
        "greedy_p80_share": g["p80_share"],
        "greedy_p60_share": g["p60_share"],
        # lp
        "lp_total_pref": l["total_pref"],
        "lp_total_assigned": l["total_flow"],
        "lp_time": try_float(lp_time),
        "lp_peak_rss_mb": try_float(lp_peak_rss),
        "lp_py_tracemalloc_peak_mb": try_float(lp_tracemalloc),
        "lp_unserved_share": l["unserved_share"],
        "lp_full_share": l["full_share"],
        "lp_p80_share": l["p80_share"],
        "lp_p60_share": l["p60_share"],
        # warm
        "warm_mcmf_total_pref": w["total_pref"],
        "warm_mcmf_total_assigned": w["total_flow"],
        "warm_mcmf_time": try_float(warm_time),
        "warm_mcmf_peak_rss_mb": try_float(warm_peak_rss),
        "warm_mcmf_py_tracemalloc_peak_mb": try_float(warm_tracemalloc),
        "warm_mcmf_unserved_count": w["unserved_count"],
        "warm_mcmf_unserved_share": w["unserved_share"],
        "warm_mcmf_full_share": w["full_share"],
        "warm_mcmf_p80_share": w["p80_share"],
        "warm_mcmf_p60_share": w["p60_share"],
        # hungarian
        "hungarian_total_pref": h["total_pref"],
        "hungarian_total_assigned": h["total_flow"],
        "hungarian_time": try_float(hungarian_time),
        "hungarian_peak_rss_mb": try_float(hungarian_peak_rss),
        "hungarian_py_tracemalloc_peak_mb": try_float(hungarian_tracemalloc),
        "hungarian_unserved_count": h["unserved_count"],
        "hungarian_unserved_share": h["unserved_share"],
        "hungarian_full_share": h["full_share"],
        "hungarian_p80_share": h["p80_share"],
        "hungarian_p60_share": h["p60_share"],
        # pso
        "pso_total_pref": p["total_pref"],
        "pso_total_assigned": p["total_flow"],
        "pso_time": try_float(pso_time),
        "pso_peak_rss_mb": try_float(pso_peak_rss),
        "pso_py_tracemalloc_peak_mb": try_float(pso_tracemalloc),
        "pso_unserved_count": p["unserved_count"],
        "pso_unserved_share": p["unserved_share"],
        "pso_full_share": p["full_share"],
        "pso_p80_share": p["p80_share"],
        "pso_p60_share": p["p60_share"],
        # meta
        "total_demand": try_float(total_demand)
    }
    return row


def try_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def try_int(x):
    if x is None:
        return None
    try:
        return int(x)
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
        df["hungarian_fulfillment"] = df.apply(
            lambda r: safe_divide(r.get("hungarian_total_assigned"), r.get("total_demand")),
            axis=1)
        df["pso_fulfillment"] = df.apply(
            lambda r: safe_divide(r.get("pso_total_assigned"), r.get("total_demand")), axis=1)
    else:
        df["mcmf_fulfillment"] = None
        df["greedy_fulfillment"] = None
        df["lp_fulfillment"] = None
        df["warm_mcmf_fulfillment"] = None
        df["hungarian_fulfillment"] = None
        df["pso_fulfillment"] = None

    # improvement percent over greedy and lp
    df["pref_greedy_vs_mcmf"] = df.apply(
        lambda r: percent_improvement(r.get("mcmf_total_pref"), r.get("greedy_total_pref")), axis=1)
    df["pref_lp_vs_mcmf"] = df.apply(lambda r: percent_improvement(r.get("mcmf_total_pref"), r.get("lp_total_pref")),
                                     axis=1)
    df["pref_warm_vs_mcmf"] = df.apply(
        lambda r: percent_improvement(r.get("mcmf_total_pref"), r.get("warm_mcmf_total_pref")), axis=1)
    df["pref_hungarian_vs_mcmf"] = df.apply(
        lambda r: percent_improvement(r.get("mcmf_total_pref"), r.get("hungarian_total_pref")), axis=1)
    df["pref_pso_vs_mcmf"] = df.apply(
        lambda r: percent_improvement(r.get("mcmf_total_pref"), r.get("pso_total_pref")), axis=1)

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

def plot_runtime_vs_pref_improved(df, plotdir):
    methods = ['mcmf', 'greedy', 'lp', 'warm_mcmf', 'hungarian', 'pso']
    colors = {"mcmf": "C0", "greedy": "C1", "lp": "C2", "warm_mcmf": "C3", "hungarian": "C4"}

    plt.figure(figsize=(10,6))
    ax = plt.gca()

    # 画每个方法的点
    for m in methods:
        xt = f"{m}_time"
        yt = f"{m}_total_pref"
        if xt in df.columns and yt in df.columns and df[xt].notnull().any() and df[yt].notnull().any():
            x = df[xt].dropna()
            y = df.loc[x.index, yt].dropna()
            # 保证索引一致
            common_idx = x.index.intersection(y.index)
            x = x.loc[common_idx].astype(float)
            y = df.loc[common_idx, yt].astype(float)

            if len(x) == 0:
                continue

            # 抖动（对数前先加小常数防止 log(0)）
            jitter = (np.random.rand(len(x)) - 0.5) * 0.05  # adjust jitter量级
            x_plot = np.array(x) + jitter

            # 使用对数 x 轴时，需要对 0 时间做小偏移
            x_plot = np.maximum(x_plot, 1e-4)

            ax.scatter(x_plot, y, label=m, color=colors.get(m, None), s=40, alpha=0.6, edgecolors='w', linewidth=0.3)

            # 标注中位数/均值
            mean_x = np.median(x)  # or np.mean(x)
            mean_y = np.median(y)
            ax.plot([mean_x], [mean_y], marker='D', markersize=6, color=colors.get(m, None), markeredgecolor='k')

    ax.set_xscale('log')  # 对数尺度
    ax.set_xlabel("Runtime (s) — log scale")
    ax.set_ylabel("Total preference")
    ax.set_title("Runtime vs Total preference (log-x, jittered)")
    ax.legend(frameon=True, fontsize='small', ncol=2)
    ax.grid(True, which='both', ls='--', lw=0.4, alpha=0.6)
    plt.tight_layout()
    fpath = os.path.join(plotdir, "runtime_vs_pref_improved_logx.png")
    plt.savefig(fpath, dpi=300)
    plt.close()
    print("Saved", fpath)


def plot_comparisons(df, plotdir, max_instances_for_bar=20):
    os.makedirs(plotdir, exist_ok=True)
    # determine methods available
    methods = ["mcmf", "greedy", "lp", "warm_mcmf", "hungarian", "pso"]
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
    if ninst <= max_instances_for_bar and pref_cols:
        ax = df[pref_cols].plot.bar(figsize=(max(8, ninst * 0.6), 6))
        ax.set_title("Total preference (higher is better) - per instance")
        ax.set_xlabel("Instance (index)")
        ax.set_ylabel("Total preference")
        plt.tight_layout()
        fpath = os.path.join(plotdir, "total_pref_per_instance.png")
        plt.savefig(fpath)
        plt.close()
        print("Saved", fpath)
    elif pref_cols:
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
    if any((c in df.columns and df[c].notnull().any()) for c in time_cols):
        # only keep cols that exist and have any non-null
        time_cols_exist = [c for c in time_cols if c in df.columns and df[c].notnull().any()]
        if ninst <= max_instances_for_bar and time_cols_exist:
            ax = df[time_cols_exist].plot.bar(figsize=(max(8, ninst * 0.6), 6))
            ax.set_title("Runtime (seconds) - per instance")
            ax.set_xlabel("Instance (index)")
            ax.set_ylabel("Time (s)")
            plt.tight_layout()
            fpath = os.path.join(plotdir, "runtime_per_instance.png")
            plt.savefig(fpath)
            plt.close()
            print("Saved", fpath)
        elif time_cols_exist:
            stats_df = df[time_cols_exist].agg(["mean", "std"]).transpose().reset_index().rename(
                columns={"index": "method"})
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

    # 3) Memory (peak RSS) comparison
    mem_cols = [f"{m}_peak_rss_mb" for m in available]
    # check existence
    if any((c in df.columns and df[c].notnull().any()) for c in mem_cols):
        mem_cols_exist = [c for c in mem_cols if c in df.columns and df[c].notnull().any()]
        if ninst <= max_instances_for_bar and mem_cols_exist:
            ax = df[mem_cols_exist].plot.bar(figsize=(max(8, ninst * 0.6), 6))
            ax.set_title("Peak memory (RSS, MB) - per instance")
            ax.set_xlabel("Instance (index)")
            ax.set_ylabel("Peak RSS (MB)")
            plt.tight_layout()
            fpath = os.path.join(plotdir, "memory_per_instance.png")
            plt.savefig(fpath)
            plt.close()
            print("Saved", fpath)
        elif mem_cols_exist:
            stats_df = df[mem_cols_exist].agg(["mean", "std"]).transpose().reset_index().rename(
                columns={"index": "method"})
            stats_df["method"] = stats_df["method"].str.replace("_peak_rss_mb", "")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(stats_df["method"], stats_df["mean"], yerr=stats_df["std"], capsize=5)
            ax.set_title("Mean peak memory (RSS MB) ± std (across instances)")
            ax.set_ylabel("Peak RSS (MB)")
            plt.tight_layout()
            fpath = os.path.join(plotdir, "memory_mean.png")
            plt.savefig(fpath)
            plt.close()
            print("Saved", fpath)

    # 4) Fulfillment rate if available
    if "mcmf_fulfillment" in df.columns and df["mcmf_fulfillment"].notnull().any():
        ful_cols = [c for c in ["mcmf_fulfillment", "greedy_fulfillment", "lp_fulfillment", "warm_mcmf_fulfillment",
                                "hungarian_fulfillment", "pso_fulfillment"] if
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

    # 5) Scatter: runtime vs total_pref for each method (if time data exists)
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False
    colors = {"mcmf": "C0", "greedy": "C1", "lp": "C2", "warm_mcmf": "C3", "hungarian": "C4"}
    for m in available:
        xt = f"{m}_time"
        yt = f"{m}_total_pref"
        if xt in df.columns and yt in df.columns and df[xt].notnull().any() and df[yt].notnull().any():
            x = df[xt]
            y = df[yt]
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

    plot_runtime_vs_pref_improved(df, plotdir)


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

    if "total_demand" in df.columns:
        df = df.sort_values(by="total_demand", key=lambda col: col.fillna(float("inf"))).reset_index(drop=True)
        print(f"Sorted {len(df)} instances by total_demand (ascending).")
    else:
        print("Warning: 'total_demand' column not available — results will not be sorted by demand.")

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
