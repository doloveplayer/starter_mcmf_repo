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
