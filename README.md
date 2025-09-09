# 资源调配

此仓库包含轻量化可运行的调配算法，包括最小费最大流、贪婪算法、LP规划器

包含如下:
- `gen_instance.py` : 生成数据 (JSON).
- `mcmf.py` : 最小费最大流算法(MCMF implementation (Successive Shortest Augmenting Path with potentials)).
- `greedy.py` : 贪婪算法.
- `lp_baseline` : 依赖 pulp轻量级 LP 求解器.
- `run_all.py` : 执行所有的调配算法, 结果存储到 `results/`.
- `analyze_results.py` : 结果分析，输出csv文件和简单的图表.
- `requirements.txt` : 依赖.

运行:
```bash
# create a test instance
python gen_instance.py --S 5 --U 12 --avg_degree 3 --out instances/test1.json --seed 42

# run solvers on that instance
python run_all.py instances/test1.json

# inspect outputs in results/
python analyze_results.py results/*.json --out summary.csv --plotdir plots --instances-dir instances

```

## Summary中的指标
- total_pref: 总偏好分（Total preference）,越大越好（表示总偏好/满意度越高）。
- total_flow: 总分配量（总流量）,成功分配的单位总数,用来衡量满足了多少总需求。
- time: 运行时间,用来衡量时效性。
- fulfillment: 总需求量,用于针对每个算法计算满足总需求量的比例。
- impr_over: 总需求量百分比提升,mcmf算法相比于基线算法LP的提升和Greedy算法的提升。

## plots
- fulfillment_per_instance: 满足率对比（assigned / demand）。用于衡量覆盖能力。
- impr_over_greedy_hist: 观察改善幅度的分布。
- runtime_per_instance: 每实例或均值的运行时间比较。用来比较算法时效性。
- runtime_vs_pref: 观察“时间—质量”权衡（例如是否存在更慢但质量更高的方法）。
- total_pref_per_instance: 直观比较每个实例上方法的偏好得分差异。

