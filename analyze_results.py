#!/usr/bin/env python3
"""
analyze_results.py (refactored, raw-image output)

主要变化：
 - 绘图改为默认输出“生图”：不显示坐标轴/刻度/标题/图例等，仅保留柱形/误差条可视内容。
 - 通过常量 CLEAN_IMAGES 控制行为（默认 True）。
 - 其余解析/统计功能与原脚本兼容。
"""
import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as path_effects
from matplotlib import font_manager

# optional scipy for paired tests
try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# -------------------------
# User-tweakable style vars
# -------------------------
TITLE_SIZE = 18
AX_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 11
BOLD_LEGEND_SIZE = 13
BOLD_WEIGHT = "heavy"  # try "heavy" or "black" if available
OUTLINE_WIDTH = 1.2
OUTLINE_COLOR = "white"
FIGURE_DPI = 300

# Whether to output "clean/raw" images (no axes, ticks, labels, legends, titles).
# Set to False if you later want full annotated plots.
CLEAN_IMAGES = True

# algorithm display order and mapping (fixed)
ALGO_ORDER = [
    "greedy",
    "lp",
    "pso",
    "mcmf",
    "warm_mcmf"
]
DISPLAY_MAP = {
    "greedy": "Greedy",
    "lp": "LP",
    "pso": "PSO",
    "mcmf": "MCMF",
    "warm_mcmf": "warm_MCMF"
}
BOLD_NAMES = {"MCMF", "warm_MCMF"}

# -------------------------
# Font management
# -------------------------
def set_mixed_fonts(simsun_path: Optional[str] = None):
    """
    Configure Matplotlib to prefer Times New Roman for Latin and SimSun for CJK.
    Optionally register simsun_path (ttf/ttc) to guarantee availability.
    Call this before creating figures.
    """
    if simsun_path:
        if os.path.isfile(simsun_path):
            try:
                font_manager.fontManager.addfont(simsun_path)
                prop = font_manager.FontProperties(fname=simsun_path)
                simsun_name = prop.get_name()
                print(f"Registered font file: {simsun_path} (internal name: {simsun_name})")
            except Exception as e:
                print("Warning: failed to register SimSun font file:", e)
        else:
            print("Warning: simsun_path provided but file not found:", simsun_path)

    preferred = ["Times New Roman", "SimSun"]
    matplotlib.rcParams['font.family'] = preferred
    matplotlib.rcParams['font.serif'] = ["Times New Roman"]
    matplotlib.rcParams['axes.unicode_minus'] = False

# call once (no path by default; if SimSun absent, matplotlib will fallback)
set_mixed_fonts(None)

# Prepare FontProperties for bold labels
try:
    fp_bold = FontProperties(family="SimHei", size=BOLD_LEGEND_SIZE, weight=BOLD_WEIGHT)
except Exception:
    fp_bold = FontProperties(size=BOLD_LEGEND_SIZE, weight=BOLD_WEIGHT)
fp_norm = FontProperties(size=LEGEND_SIZE)

# helper to emphasize legend labels (bold + outline)
def emphasize_legend_entries(leg, emphasize_names=set()):
    if leg is None:
        return
    for text in leg.get_texts():
        text.set_fontproperties(fp_norm)
    for text in leg.get_texts():
        txt = text.get_text()
        if txt in emphasize_names:
            text.set_fontproperties(fp_bold)
            text.set_fontweight(BOLD_WEIGHT)
            text.set_path_effects([
                path_effects.Stroke(linewidth=OUTLINE_WIDTH, foreground=OUTLINE_COLOR),
                path_effects.Normal()
            ])

def save_csv(df, out_path):
    df.to_csv(out_path, index=False)
    print(f"Wrote CSV summary to {out_path}")

# -------------------------
# Data parsing utilities (unchanged logic)
# -------------------------
def load_instance_user_needs(instances_dir, instance_name):
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
    if instances_dir is None:
        return None
    p = Path(instances_dir) / f"{instance_name}.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            inst = json.load(f)
        total = 0.0
        for u in inst.get("users", []):
            try:
                total += float(u.get("need", 0))
            except Exception:
                try:
                    total += float(str(u.get("need", 0)))
                except Exception:
                    pass
        return total
    except Exception:
        return None

def load_instance_score_map(instances_dir, instance_name):
    if instances_dir is None:
        return None
    p = Path(instances_dir) / f"{instance_name}.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            inst = json.load(f)
        score_map = {}
        for u in inst.get("users", []):
            uid = u.get("id")
            if uid is None:
                continue
            prefs = u.get("supplier_scores", [])
            for entry in prefs:
                try:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        sid = entry[0]
                        sc = entry[1]
                    elif isinstance(entry, dict):
                        sid = entry.get("supplier") or entry.get("sid") or entry.get("id")
                        sc = entry.get("score") or entry.get("sc") or entry.get("val")
                    else:
                        continue
                    if sid is None or sc is None:
                        continue
                    score_map[(str(uid), str(sid))] = float(sc)
                except Exception:
                    continue
        return score_map
    except Exception:
        return None

def extract_alloc_map(result_dict):
    if not isinstance(result_dict, dict):
        return None, None
    allocs = result_dict.get("allocations") or result_dict.get("allocation") or result_dict.get("alloc") or None
    if allocs is None:
        for k, v in result_dict.items():
            if isinstance(v, dict):
                sample_vals = list(v.values())[:3]
                if sample_vals and all(isinstance(x, (list, tuple, dict)) for x in sample_vals):
                    allocs = v
                    break
        if allocs is None:
            return None, None
    if isinstance(allocs, list):
        alloc_map = {}
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

    out_map = {}
    for uid, lst in alloc_map.items():
        if not lst:
            out_map[str(uid)] = 0.0
            continue
        if isinstance(lst, (int, float)):
            out_map[str(uid)] = float(lst)
            continue
        total = 0.0
        for entry in lst:
            amt = 0.0
            if entry is None:
                continue
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                try:
                    amt = float(entry[1])
                except Exception:
                    try:
                        amt = float(str(entry[1]))
                    except Exception:
                        amt = 0.0
            elif isinstance(entry, dict):
                found = False
                for key in ("amt", "amount", "assigned", "quantity", "flow", "val"):
                    if key in entry:
                        try:
                            amt = float(entry[key])
                            found = True
                            break
                        except Exception:
                            amt = 0.0
                if not found:
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

def extract_alloc_full(result_dict):
    if not isinstance(result_dict, dict):
        return None
    allocs = result_dict.get("allocations") or result_dict.get("allocation") or result_dict.get("alloc") or None
    if allocs is None:
        for k, v in result_dict.items():
            if isinstance(v, dict):
                sample_vals = list(v.values())[:3]
                if sample_vals and all(isinstance(x, (list, tuple, dict)) for x in sample_vals):
                    allocs = v
                    break
        if allocs is None:
            return None
    alloc_full = {}
    try:
        if isinstance(allocs, dict):
            for uid, lst in allocs.items():
                uid_s = str(uid)
                parsed = []
                if lst is None:
                    alloc_full[uid_s] = []
                    continue
                if isinstance(lst, (int, float)):
                    alloc_full[uid_s] = [("UNKNOWN", float(lst))]
                    continue
                for entry in lst:
                    if entry is None:
                        continue
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        try:
                            sid = entry[0]
                            amt = float(entry[1])
                            parsed.append((str(sid), amt))
                        except Exception:
                            continue
                    elif isinstance(entry, dict):
                        sid = None
                        amt = None
                        for key in ("supplier", "sid", "sup", "s"):
                            if key in entry:
                                sid = entry.get(key)
                                break
                        for key in ("amt", "amount", "assigned", "quantity", "flow", "val"):
                            if key in entry:
                                try:
                                    amt = float(entry.get(key))
                                except Exception:
                                    amt = None
                                break
                        if sid is not None and amt is not None:
                            parsed.append((str(sid), amt))
                        else:
                            vals = list(entry.values())
                            if len(vals) >= 2 and isinstance(vals[1], (int, float, str)):
                                try:
                                    amt = float(vals[1])
                                    sid = vals[0]
                                    parsed.append((str(sid), amt))
                                except Exception:
                                    continue
                    else:
                        try:
                            amt = float(entry)
                            parsed.append(("UNKNOWN", amt))
                        except Exception:
                            continue
                alloc_full[uid_s] = parsed
        elif isinstance(allocs, list):
            for item in allocs:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    uid = str(item[0])
                    lst = item[1]
                    parsed = []
                    if isinstance(lst, (int, float)):
                        parsed = [("UNKNOWN", float(lst))]
                    else:
                        for entry in lst:
                            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                                try:
                                    sid = entry[0]
                                    amt = float(entry[1])
                                    parsed.append((str(sid), amt))
                                except Exception:
                                    continue
                            elif isinstance(entry, dict):
                                sid = None
                                amt = None
                                for key in ("supplier", "sid", "sup", "s"):
                                    if key in entry:
                                        sid = entry.get(key)
                                        break
                                for key in ("amt", "amount", "assigned", "quantity", "flow", "val"):
                                    if key in entry:
                                        try:
                                            amt = float(entry.get(key))
                                        except Exception:
                                            amt = None
                                        break
                                if sid is not None and amt is not None:
                                    parsed.append((str(sid), amt))
                    alloc_full[uid] = parsed
                elif isinstance(item, dict):
                    uid = item.get("user") or item.get("uid") or item.get("id")
                    lst = item.get("allocations") or item.get("alloc") or item.get("assigned") or item.get("allocation")
                    if uid is None or lst is None:
                        continue
                    uid_s = str(uid)
                    parsed = []
                    if isinstance(lst, (int, float)):
                        parsed = [("UNKNOWN", float(lst))]
                    else:
                        for entry in lst:
                            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                                try:
                                    sid = entry[0]
                                    amt = float(entry[1])
                                    parsed.append((str(sid), amt))
                                except Exception:
                                    continue
                            elif isinstance(entry, dict):
                                sid = None
                                amt = None
                                for key in ("supplier", "sid", "sup", "s"):
                                    if key in entry:
                                        sid = entry.get(key)
                                        break
                                for key in ("amt", "amount", "assigned", "quantity", "flow", "val"):
                                    if key in entry:
                                        try:
                                            amt = float(entry.get(key))
                                        except Exception:
                                            amt = None
                                        break
                                if sid is not None and amt is not None:
                                    parsed.append((str(sid), amt))
                    alloc_full[uid_s] = parsed
        else:
            return None
    except Exception:
        return None
    return alloc_full

def compute_pref_band_shares(alloc_full, score_map):
    if alloc_full is None or score_map is None:
        return {
            "total_assigned": None,
            "gt90_amt": None, "70_90_amt": None, "lt70_amt": None,
            "gt90_share": None, "70_90_share": None, "lt70_share": None
        }
    total = gt90 = b70_90 = lt70 = 0.0
    for uid, lst in alloc_full.items():
        for sid, amt in lst:
            try:
                a = float(amt)
            except Exception:
                continue
            total += a
            sc = score_map.get((str(uid), str(sid)))
            if sc is None:
                sc = score_map.get((str(uid), str(sid)))
            if sc is None:
                continue
            try:
                scv = float(sc)
            except Exception:
                continue
            if scv > 90.0:
                gt90 += a
            elif scv >= 70.0:
                b70_90 += a
            else:
                lt70 += a
    if total <= 0:
        return {
            "total_assigned": float(total),
            "gt90_amt": float(gt90), "70_90_amt": float(b70_90), "lt70_amt": float(lt70),
            "gt90_share": None if total == 0 else float(gt90 / total),
            "70_90_share": None if total == 0 else float(b70_90 / total),
            "lt70_share": None if total == 0 else float(lt70 / total)
        }
    return {
        "total_assigned": float(total),
        "gt90_amt": float(gt90), "70_90_amt": float(b70_90), "lt70_amt": float(lt70),
        "gt90_share": float(gt90 / total), "70_90_share": float(b70_90 / total), "lt70_share": float(lt70 / total)
    }

def compute_satisfaction_stats(alloc_map, user_needs_map, total_users_from_meta=None):
    if alloc_map is None:
        return {
            "full_count": None, "full_share": None,
            "p80_count": None, "p80_share": None,
            "p60_count": None, "p60_share": None,
            "total_users_in_alloc": None
        }
    if user_needs_map is None:
        return {
            "full_count": None, "full_share": None,
            "p80_count": None, "p80_share": None,
            "p60_count": None, "p60_share": None,
            "total_users_in_alloc": len(alloc_map)
        }
    full = p80 = p60 = 0
    total_users = 0
    for uid, alloc in alloc_map.items():
        total_users += 1
        need = user_needs_map.get(str(uid))
        if need is None or need == 0:
            continue
        ratio = float(alloc) / float(need)
        if ratio >= 1.0:
            full += 1
        if ratio >= 0.8:
            p80 += 1
        if ratio >= 0.6:
            p60 += 1
    denom = None
    if total_users_from_meta is not None:
        try:
            if int(total_users_from_meta) > 0:
                denom = int(total_users_from_meta)
        except Exception:
            denom = None
    if denom is None:
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

# -------------------------
# result parsing (per-file -> row)
# -------------------------
def parse_result_file(path: str, instances_dir: Optional[str] = None) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    name = data.get("instance", Path(path).stem)
    meta = data.get("meta", {})

    user_needs_map = load_instance_user_needs(instances_dir, name)
    score_map = load_instance_score_map(instances_dir, name)

    def get_nested(d, *keys, default=None):
        cur = d
        for k in keys:
            if cur is None:
                return default
            cur = cur.get(k, None)
        return cur if cur is not None else default

    def analyze_method(result_dict):
        alloc_map, total_users_in_alloc = extract_alloc_map(result_dict)
        alloc_full = extract_alloc_full(result_dict)

        if alloc_map is None:
            unserved_count = None
            unserved_share = None
        else:
            unserved_count = sum(1 for v in alloc_map.values() if (v is None or float(v) <= 0.0))
            denom = None
            try:
                denom = int(meta.get("U")) if meta.get("U") is not None else None
            except Exception:
                denom = None
            if denom is None:
                denom = total_users_in_alloc if total_users_in_alloc > 0 else None
            unserved_share = None if denom is None else float(unserved_count) / float(denom)

        sat_stats = compute_satisfaction_stats(alloc_map, user_needs_map, total_users_from_meta=meta.get("U"))

        total_pref = None
        total_flow = None
        if isinstance(result_dict, dict):
            total_pref = result_dict.get("total_pref_score") or result_dict.get("total_pref") or result_dict.get("pref") or None
            total_flow = result_dict.get("total_flow") or result_dict.get("total_assigned") or result_dict.get("assigned") or None

        band_stats = compute_pref_band_shares(alloc_full, score_map) if alloc_full is not None else {
            "total_assigned": None, "gt90_amt": None, "70_90_amt": None, "lt70_amt": None,
            "gt90_share": None, "70_90_share": None, "lt70_share": None
        }

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
            "total_flow": try_float(total_flow),
            "band_total_assigned": try_float(band_stats["total_assigned"]),
            "band_gt90_amt": try_float(band_stats["gt90_amt"]),
            "band_70_90_amt": try_float(band_stats["70_90_amt"]),
            "band_lt70_amt": try_float(band_stats["lt70_amt"]),
            "band_gt90_share": try_float(band_stats["gt90_share"]),
            "band_70_90_share": try_float(band_stats["70_90_share"]),
            "band_lt70_share": try_float(band_stats["lt70_share"])
        }

    # parse each method
    mcmf_result = get_nested(data, "mcmf", "result", default=None)
    mcmf_time = get_nested(data, "mcmf", "time", default=None) or get_nested(data, "mcmf", "timings", "total_time", default=None)
    mcmf_peak_rss = get_nested(data, "mcmf", "peak_rss_mb", default=None)
    mcmf_tracemalloc = get_nested(data, "mcmf", "py_tracemalloc_peak_mb", default=None)
    m = analyze_method(mcmf_result)

    greedy_result = get_nested(data, "greedy", "result", default=None)
    greedy_time = get_nested(data, "greedy", "time", default=None) or get_nested(data, "greedy", "timings", "total_time", default=None)
    greedy_peak_rss = get_nested(data, "greedy", "peak_rss_mb", default=None)
    greedy_tracemalloc = get_nested(data, "greedy", "py_tracemalloc_peak_mb", default=None)
    g = analyze_method(greedy_result)

    lp_result = get_nested(data, "lp", "result", default=None)
    lp_time = get_nested(data, "lp", "time", default=None) or get_nested(data, "lp", "timings", "total_time", default=None)
    lp_peak_rss = get_nested(data, "lp", "peak_rss_mb", default=None)
    lp_tracemalloc = get_nested(data, "lp", "py_tracemalloc_peak_mb", default=None)
    l = analyze_method(lp_result)

    warm_result = get_nested(data, "warm_mcmf", "result", default=None)
    warm_time = get_nested(data, "warm_mcmf", "time", default=None) or get_nested(data, "warm_mcmf", "timings", "total_time", default=None)
    warm_peak_rss = get_nested(data, "warm_mcmf", "peak_rss_mb", default=None)
    warm_tracemalloc = get_nested(data, "warm_mcmf", "py_tracemalloc_peak_mb", default=None)
    w = analyze_method(warm_result)

    hungarian_result = get_nested(data, "hungarian", "result", default=None)
    hungarian_time = get_nested(data, "hungarian", "time", default=None) or get_nested(data, "hungarian", "timings", "total_time", default=None)
    hungarian_peak_rss = get_nested(data, "hungarian", "peak_rss_mb", default=None)
    hungarian_tracemalloc = get_nested(data, "hungarian", "py_tracemalloc_peak_mb", default=None)
    h = analyze_method(hungarian_result)

    pso_result = get_nested(data, "pso", "result", default=None)
    pso_time = get_nested(data, "pso", "time", default=None) or get_nested(data, "pso", "timings", "total_time", default=None)
    pso_peak_rss = get_nested(data, "pso", "peak_rss_mb", default=None)
    pso_tracemalloc = get_nested(data, "pso", "py_tracemalloc_peak_mb", default=None)
    p = analyze_method(pso_result)

    total_demand = None
    if instances_dir is not None:
        total_demand = load_instance_total_demand(instances_dir, name)

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
        "mcmf_pref_gt90_share": m["band_gt90_share"],
        "mcmf_pref_70_90_share": m["band_70_90_share"],
        "mcmf_pref_lt70_share": m["band_lt70_share"],
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
        "greedy_pref_gt90_share": g["band_gt90_share"],
        "greedy_pref_70_90_share": g["band_70_90_share"],
        "greedy_pref_lt70_share": g["band_lt70_share"],
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
        "lp_pref_gt90_share": l["band_gt90_share"],
        "lp_pref_70_90_share": l["band_70_90_share"],
        "lp_pref_lt70_share": l["band_lt70_share"],
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
        "warm_pref_gt90_share": w["band_gt90_share"],
        "warm_pref_70_90_share": w["band_70_90_share"],
        "warm_pref_lt70_share": w["band_lt70_share"],
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
        "hungarian_pref_gt90_share": h["band_gt90_share"],
        "hungarian_pref_70_90_share": h["band_70_90_share"],
        "hungarian_pref_lt70_share": h["band_lt70_share"],
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
        "pso_pref_gt90_share": p["band_gt90_share"],
        "pso_pref_70_90_share": p["band_70_90_share"],
        "pso_pref_lt70_share": p["band_lt70_share"],
        # meta
        "total_demand": try_float(total_demand)
    }
    return row

# -------------------------
# small helpers
# -------------------------
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
    if "total_demand" in df.columns:
        df["mcmf_fulfillment"] = df.apply(lambda r: safe_divide(r.get("mcmf_total_flow"), r.get("total_demand")), axis=1)
        df["greedy_fulfillment"] = df.apply(lambda r: safe_divide(r.get("greedy_total_assigned"), r.get("total_demand")), axis=1)
        df["lp_fulfillment"] = df.apply(lambda r: safe_divide(r.get("lp_total_assigned"), r.get("total_demand")), axis=1)
        df["warm_mcmf_fulfillment"] = df.apply(lambda r: safe_divide(r.get("warm_mcmf_total_assigned"), r.get("total_demand")), axis=1)
        df["pso_fulfillment"] = df.apply(lambda r: safe_divide(r.get("pso_total_assigned"), r.get("total_demand")), axis=1)
    else:
        df["mcmf_fulfillment"] = None
        df["greedy_fulfillment"] = None
        df["lp_fulfillment"] = None
        df["warm_mcmf_fulfillment"] = None
        df["pso_fulfillment"] = None

    df["pref_greedy_vs_mcmf"] = df.apply(lambda r: percent_improvement(r.get("mcmf_total_pref"), r.get("greedy_total_pref")), axis=1)
    df["pref_lp_vs_mcmf"] = df.apply(lambda r: percent_improvement(r.get("mcmf_total_pref"), r.get("lp_total_pref")), axis=1)
    df["pref_warm_vs_mcmf"] = df.apply(lambda r: percent_improvement(r.get("mcmf_total_pref"), r.get("warm_mcmf_total_pref")), axis=1)
    df["pref_hungarian_vs_mcmf"] = df.apply(lambda r: percent_improvement(r.get("mcmf_total_pref"), r.get("hungarian_total_pref")), axis=1)
    df["pref_pso_vs_mcmf"] = df.apply(lambda r: percent_improvement(r.get("mcmf_total_pref"), r.get("pso_total_pref")), axis=1)
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
    if a is None or b is None:
        return None
    denom = max(abs(b), 1e-9)
    return 100.0 * (a - b) / denom

# -------------------------
# plotting helpers (clean/raw output)
# -------------------------
def _finalize_and_save(ax, plotdir, fname, dpi=FIGURE_DPI):
    """
    Helper: if CLEAN_IMAGES True -> remove axes/ticks/legend/title/labels, save tightly with zero padding.
    """
    os.makedirs(plotdir, exist_ok=True)
    fpath = os.path.join(plotdir, fname)
    if CLEAN_IMAGES:
        # hide axes, ticks, spines, legend, title
        ax.set_frame_on(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # remove spines
        for sp in ax.spines.values():
            sp.set_visible(False)
        # remove legend if any
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        # remove title
        ax.set_title("")
        # save tightly with no padding
        plt.savefig(fpath, dpi=dpi, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(fpath, dpi=dpi)
    plt.close()
    print("Saved", fpath)
    return fpath

def plot_grouped_bars_per_instance(df, cols, display_names, plot_title, ylabel, plotdir, fname):
    """
    df: full dataframe
    cols: list of column names present in df (ordered)
    display_names: list of friendly labels (same length)
    Outputs clean/raw image by default (CLEAN_IMAGES=True).
    """
    ninst = len(df)
    if ninst == 0 or not cols:
        return None
    figsize = (max(8, ninst * 0.5), 6) if ninst <= 60 else (14, 6)
    fig, ax = plt.subplots(figsize=figsize)
    # draw bars: use numpy array to preserve order
    arr = df[cols].to_numpy()
    # let pandas do grouped bar drawing for convenience but on our axes
    # we construct a temporary DataFrame to use plot.bar on our axes
    tmp = pd.DataFrame(data=arr, columns=display_names)
    tmp.plot.bar(ax=ax, legend=not CLEAN_IMAGES)  # show legend only when not clean

    # 1-based x ticks if not cleaning (for raw we remove)
    if not CLEAN_IMAGES:
        ax.set_xticks(range(ninst))
        if ninst > 30:
            rot, ha = 45, "right"
        elif ninst > 10:
            rot, ha = 30, "right"
        else:
            rot, ha = 0, "center"
        ax.set_xticklabels([str(i) for i in range(1, ninst + 1)], rotation=rot, ha=ha)
        ax.tick_params(axis='x', labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)
        ax.set_xlabel("实例", fontsize=AX_LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=AX_LABEL_SIZE)
        ax.set_title(plot_title, fontsize=TITLE_SIZE)
        # legend handling
        leg = ax.legend(fontsize=LEGEND_SIZE, frameon=True)
        emphasize_legend_entries(leg, emphasize_names=BOLD_NAMES)
    else:
        # when clean: remove tick labels (we already hide in finalize helper)
        pass

    return _finalize_and_save(ax, plotdir, fname)

def plot_mean_bars(df, cols, display_names, plot_title, ylabel, plotdir, fname):
    """
    Mean ± std bar plot across instances. Produces clean/raw image by default.
    """
    stats_df = df[cols].agg(["mean", "std"]).transpose().reset_index().rename(columns={"index": "method"})
    # method column contains original col names; map to display_names
    stats_df["display"] = [DISPLAY_MAP.get(m.replace("_total_pref","").replace("_time","").replace("_peak_rss_mb","").replace("_fulfillment",""), m) for m in stats_df["method"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(stats_df))
    means = stats_df["mean"].to_numpy(dtype=float)
    errs = stats_df["std"].to_numpy(dtype=float)
    bars = ax.bar(x, means, yerr=errs, capsize=5)
    if not CLEAN_IMAGES:
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df["display"], rotation=30, ha='right', fontsize=TICK_LABEL_SIZE)
        for lbl in ax.get_xticklabels():
            if lbl.get_text() in BOLD_NAMES:
                lbl.set_fontweight("bold")
                lbl.set_fontsize(BOLD_LEGEND_SIZE)
                lbl.set_path_effects([path_effects.Stroke(linewidth=OUTLINE_WIDTH, foreground=OUTLINE_COLOR), path_effects.Normal()])
        ax.set_title(plot_title, fontsize=TITLE_SIZE)
        ax.set_ylabel(ylabel, fontsize=AX_LABEL_SIZE)
    return _finalize_and_save(ax, plotdir, fname)

# -------------------------
# Main plotting orchestration
# -------------------------
def plot_comparisons(df: pd.DataFrame, plotdir: str, max_instances_for_bar: int = 20):
    os.makedirs(plotdir, exist_ok=True)
    ninst = len(df)
    if ninst == 0:
        print("No instances to plot.")
        return

    # prepare columns in display order (ALGO_ORDER)
    # 1) Total preference
    pref_cols = [f"{m}_total_pref" for m in ALGO_ORDER if f"{m}_total_pref" in df.columns and df[f"{m}_total_pref"].notnull().any()]
    pref_display = [DISPLAY_MAP.get(m, m) for m in [c.replace("_total_pref", "") for c in pref_cols]]
    if pref_cols:
        if ninst <= max_instances_for_bar:
            plot_grouped_bars_per_instance(df, pref_cols, pref_display, "Total preference (higher is better) - per instance", "Total preference", plotdir, "total_pref_per_instance.png")
        else:
            plot_mean_bars(df, pref_cols, pref_display, "Mean total preference ± std (across instances)", "Total preference", plotdir, "total_pref_mean.png")

    # 2) Runtime (exclude pso)
    runtime_cols = [f"{m}_time" for m in ALGO_ORDER if m != "pso" and f"{m}_time" in df.columns and df[f"{m}_time"].notnull().any()]
    runtime_display = [DISPLAY_MAP.get(m, m) for m in [c.replace("_time", "") for c in runtime_cols]]
    if runtime_cols:
        if ninst <= max_instances_for_bar:
            plot_grouped_bars_per_instance(df, runtime_cols, runtime_display, "Runtime (seconds) - per instance (pso excluded)", "Time (s)", plotdir, "runtime_per_instance.png")
        else:
            plot_mean_bars(df, runtime_cols, runtime_display, "Mean runtime ± std (across instances) (pso excluded)", "Time (s)", plotdir, "runtime_mean.png")

    # 3) Memory
    mem_cols = [f"{m}_peak_rss_mb" for m in ALGO_ORDER if f"{m}_peak_rss_mb" in df.columns and df[f"{m}_peak_rss_mb"].notnull().any()]
    mem_display = [DISPLAY_MAP.get(m, m) for m in [c.replace("_peak_rss_mb", "") for c in mem_cols]]
    if mem_cols:
        if ninst <= max_instances_for_bar:
            plot_grouped_bars_per_instance(df, mem_cols, mem_display, "Peak memory (RSS, MB) - per instance", "Peak RSS (MB)", plotdir, "memory_per_instance.png")
        else:
            plot_mean_bars(df, mem_cols, mem_display, "Mean peak memory (RSS MB) ± std (across instances)", "Peak RSS (MB)", plotdir, "memory_mean.png")

    # 4) Fulfillment
    ful_cols = [f"{m}_fulfillment" for m in ALGO_ORDER if f"{m}_fulfillment" in df.columns and df[f"{m}_fulfillment"].notnull().any()]
    ful_display = [DISPLAY_MAP.get(m, m) for m in [c.replace("_fulfillment", "") for c in ful_cols]]
    if ful_cols:
        if ninst <= max_instances_for_bar:
            plot_grouped_bars_per_instance(df, ful_cols, ful_display, "Fulfillment rate (assigned / total demand) - per instance", "Fulfillment", plotdir, "fulfillment_per_instance.png")
        else:
            plot_mean_bars(df, ful_cols, ful_display, "Mean fulfillment rate ± std (across instances)", "Fulfillment rate", plotdir, "fulfillment_mean.png")

    print("Plotting complete.")

# -------------------------
# Statistical tests helper
# -------------------------
def statistical_tests(df):
    res = {}
    if not HAS_SCIPY:
        print("scipy not available: skipping statistical tests")
        return res
    a = df.get("mcmf_total_pref")
    b = df.get("greedy_total_pref")
    if a is None or b is None:
        return res
    mask = a.notnull() & b.notnull()
    if mask.any():
        t, p = stats.ttest_rel(a[mask], b[mask])
        res["mcmf_vs_greedy_t"] = float(t)
        res["mcmf_vs_greedy_p"] = float(p)
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
            res["mcmf_vs_warm_t"] = float(t2)
            res["mcmf_vs_warm_p"] = float(p2)
    return res

# -------------------------
# CLI and main
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Aggregate JSON result files and produce CSV + raw-image plots")
    p.add_argument("results", nargs="+", help="one or more result JSON files or glob pattern (e.g. results/*.json)")
    p.add_argument("--out", default="summary.csv", help="CSV output file")
    p.add_argument("--plotdir", default="plots", help="directory to save plots")
    p.add_argument("--instances-dir", default=None, help="optional directory containing instance JSONs")
    p.add_argument("--simsun", default=None, help="optional path to simsun.ttf/ttc to register for Chinese rendering")
    p.add_argument("--max-instances-for-bar", type=int, default=20, help="max instances to draw per-instance grouped bars")
    p.add_argument("--no-clean", action="store_true", help="if set, produce annotated plots (turn CLEAN_IMAGES=False)")
    args = p.parse_args()

    global CLEAN_IMAGES
    if args.simsun:
        set_mixed_fonts(args.simsun)
    if args.no_clean:
        CLEAN_IMAGES = False

    # expand globs and directories
    paths = []
    for r in args.results:
        if "*" in r or "?" in r or "[" in r:
            paths.extend(glob.glob(r))
        elif os.path.isdir(r):
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
    print("\nSummary statistics (means across instances):")
    summary = df.mean(numeric_only=True).to_dict()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    stats_res = statistical_tests(df)
    if stats_res:
        print("\nStatistical test results:")
        for k, v in stats_res.items():
            print(f"  {k}: {v}")

    plot_comparisons(df, args.plotdir, max_instances_for_bar=args.max_instances_for_bar)

if __name__ == "__main__":
    main()
