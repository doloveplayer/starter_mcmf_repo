#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hungarian.py

匈牙利算法（Kuhn-Munkres）实现 + 单位扩展封装，用于把 assignment（one-to-one）扩展到
supplier stock / user need 的离散分配场景，方便与 MCMF / LP 做对比。

主要函数：
  - hungarian(cost_matrix): 对方阵 cost_matrix 求解最小化总成本的完美匹配（返回 match 数组）
  - hungarian_allocation(inst, max_expand=50000, default_score=0, verbose=False):
        将实例扩展为单位层面的 assignment，调用匈牙利，聚合结果并返回与其它方法兼容的字典格式：
        {
          "total_flow": int,
          "total_pref_score": int,
          "allocations": { user_id: [(supplier_id, amt), ...], ... },
          "timings": {"hungarian_time": float},
        }

用法示例：
  from hungarian import hungarian_allocation
  res = hungarian_allocation(instance_dict, max_expand=20000, verbose=True)

注意：
  - 这是纯 Python 实现，时间复杂度 O(n^3)。对于扩展后单位数 n>2000 的实例，运行时间可能会很长。
  - 如果你的 score 最大值已知（例如 100），可以传入 default_score 参数或在实例中保证 score 范围。
"""
import time
from pathlib import Path

def hungarian(cost):
    """
    Hungarian (Kuhn-Munkres) 算法 — 最小化 cost 总和（方阵）。
    cost: n x n 的二维 list，cost[i][j] 为 i->j 的成本（数值）
    返回 match_r: 长度 n 的数组，match_r[j] = i 表示行 i 匹配到列 j，
            若列 j 未被任何行匹配（理论上不会发生在方阵完美匹配时），会为 -1。
    参考实现：匈牙利标准带潜变量做法（基于 emaxx.ru 的思路）
    """
    n = len(cost)
    if n == 0:
        return []

    # u,v 为顶标 (potential)，p,way 为匹配辅助数组
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)      # p[j] = 配到列 j 的行索引
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            # 寻找列 j1，使得 minv[j1] 减少
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            # 对顶标进行调整
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            # 如果列 j0 未被匹配，结束
            if p[j0] == 0:
                break
        # 反向构造增广路径
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # p[j] 表示列 j 被 p[j] 行匹配，p[0] 为 dummy
    match_r = [-1] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            match_r[j - 1] = p[j] - 1
    return match_r


def hungarian_allocation(inst, max_expand=50000, default_score=0, verbose=False):
    """
    对给定实例使用匈牙利算法做对比分配（单位展开策略）。
    参数：
      - inst: instance dict, structure:
          inst["suppliers"]: list of {"id": sid, "stock": int, ...}
          inst["users"]: list of {"id": uid, "need": int, "supplier_scores": [(sid, score), ...], ...}
      - max_expand: 最大允许的单位节点数（supplier_units + user_units），超出则抛错以避免内存/时间爆炸
      - default_score: 若某 (s,u) pair 在 supplier_scores 中缺失，则使用该默认分数（通常 0）
      - verbose: 是否打印进度信息
    返回：
      {
        "total_flow": int (匹配到的单位数),
        "total_pref_score": int,
        "allocations": { user_id: [(supplier_id, amt), ...], ... },
        "timings": {"hungarian_time": float}
      }
    注意：本实现对规模较小的实例（展开单位总数 < ~5000）最为实用。
    """
    t0 = time.time()

    suppliers = inst.get("suppliers", [])
    users = inst.get("users", [])

    # 构建 score map: (s_id, u_id) -> score (int)
    score_map = {}
    max_score = default_score
    for u in users:
        for sid, sc in u.get("supplier_scores", []):
            try:
                scv = int(sc)
            except Exception:
                try:
                    scv = int(float(sc))
                except Exception:
                    scv = default_score
            score_map[(str(sid), str(u["id"]))] = scv
            if scv > max_score:
                max_score = scv

    # 扩展 supplier 单位与 user 单位
    sup_unit_to_sid = []  # index -> supplier id (string)
    for s in suppliers:
        sid = str(s["id"])
        stock = int(s.get("stock", 0))
        for _ in range(stock):
            sup_unit_to_sid.append(sid)

    user_unit_to_uid = []
    for u in users:
        uid = str(u["id"])
        need = int(u.get("need", 0))
        for _ in range(need):
            user_unit_to_uid.append(uid)

    m = len(sup_unit_to_sid)
    n = len(user_unit_to_uid)
    if verbose:
        print(f"[hungarian] supplier_units={m}, user_units={n}, total_units={m+n}")
    if m + n > max_expand:
        raise ValueError(f"Expanded unit count (m+n={m+n}) exceeds max_expand={max_expand}.")

    if m == 0 or n == 0:
        # 没有供给或需求，直接返回空结果
        return {
            "total_flow": 0,
            "total_pref_score": 0,
            "allocations": {str(u["id"]): [] for u in users},
            "timings": {"hungarian_time": time.time() - t0}
        }

    # 匈牙利需要方阵，取 size = max(m,n)
    size = max(m, n)

    # 构造 cost 矩阵 (size x size)，使得最小化 cost 等价于最大化 score
    # cost = (max_score + 1) - score for existing edges（确保成本非负）
    # 对于不存在的 (s,u) pair，我们设成本非常大（INF）
    base = max_score + 1
    cost = [[base + 0 for _ in range(size)] for __ in range(size)]  # 默认较大 cost

    # 填充实际 (supplier_unit i -> user_unit j) 的 cost
    for i in range(m):
        sid = sup_unit_to_sid[i]
        for j in range(n):
            uid = user_unit_to_uid[j]
            sc = score_map.get((sid, uid))
            if sc is None:
                sc = default_score  # 若没有评分，就用默认（通常 0）
            # 映射到成本（越高的 score -> 越低的 cost）
            cost[i][j] = base - int(sc)

    # 剩余填充（dummy 行/列）保持较大成本 base (或 INF) 以避免匹配到虚拟单元
    # 运行匈牙利
    match_r = hungarian(cost)

    # match_r 长度 = size（列 -> 行），match_r[j] = i 表示行 i 匹配列 j
    # 我们关注真实的 i<m 且 j<n 的配对
    allocations = {}  # user_id -> dict supplier_id -> count
    total_flow = 0
    total_pref = 0
    for j in range(n):  # 对真实用户单位 j
        i = match_r[j]
        if i is None or i < 0:
            continue
        if i >= m:
            # 被分配到了 dummy supplier 单位 -> 表示该用户单位未被分配
            continue
        # i (supplier unit) matched to j (user unit)
        sid = sup_unit_to_sid[i]
        uid = user_unit_to_uid[j]
        allocations.setdefault(uid, {})
        allocations[uid][sid] = allocations[uid].get(sid, 0) + 1
        total_flow += 1
        # 累加偏好得分（使用原始 score）
        sc = score_map.get((sid, uid), default_score)
        total_pref += int(sc)

    # 把 allocations 转为 list-of-tuples 格式以兼容你现有的结果 schema
    alloc_out = {}
    for u in users:
        uid = str(u["id"])
        if uid not in allocations:
            alloc_out[uid] = []
        else:
            alloc_out[uid] = [(sid, amt) for sid, amt in allocations[uid].items()]

    hung_time = time.time() - t0

    res = {
        "total_flow": int(total_flow),
        "total_pref_score": int(total_pref),
        "allocations": alloc_out,
        "timings": {"hungarian_time": hung_time}
    }
    if verbose:
        print(f"[hungarian] matched units={total_flow}, total_pref={total_pref}, time={hung_time:.4f}s")
    return res


# 简单的命令行接口，供直接调试 / 小规模试验
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--inst", required=True, help="instance json path")
    parser.add_argument("--out", default=None, help="output json path")
    parser.add_argument("--max-expand", type=int, default=5000, help="max expansion limit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.inst, "r", encoding="utf-8") as f:
        inst = json.load(f)

    res = hungarian_allocation(inst, max_expand=args.max_expand, verbose=args.verbose)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            json.dump(res, fo, indent=2, ensure_ascii=False)
    else:
        import pprint
        pprint.pprint(res)
