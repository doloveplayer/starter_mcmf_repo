#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mcmf_visualizer.py

MCMF网络流可视化工具
用于生成论文插图的网络流图，显示源点、汇点、用户和供应商节点。
可以通过参数控制显示的节点数量，避免图像过于拥挤。
"""

import argparse
import json
import matplotlib.pyplot as plt
import networkx as nx
from mcmf import MinCostMaxFlow
import matplotlib.patches as mpatches


class MCMFVisualizer:
    def __init__(self, inst, max_nodes=20, show_flow=True):
        """
        初始化可视化器

        参数:
            inst: 实例数据
            max_nodes: 最大显示的节点数量
            show_flow: 是否显示流量
        """
        self.inst = inst
        self.max_nodes = max_nodes
        self.show_flow = show_flow
        self.G = nx.DiGraph()
        self.pos = {}
        self.node_colors = []
        self.edge_labels = {}

    def create_graph(self):
        """从实例创建网络流图"""
        suppliers = self.inst.get("suppliers", [])
        users = self.inst.get("users", [])

        # 限制显示的节点数量
        if len(suppliers) + len(users) + 2 > self.max_nodes:
            print(f"警告: 实例包含 {len(suppliers) + len(users) + 2} 个节点，超过最大显示限制 {self.max_nodes}")
            print("将显示前几个供应商和用户节点")

            # 选择部分供应商和用户
            suppliers = suppliers[:max(1, self.max_nodes // 2 - 1)]
            users = users[:max(1, self.max_nodes - len(suppliers) - 2)]

        S = len(suppliers)
        U = len(users)

        # 添加节点
        self.G.add_node("s", type="source")  # 源点
        self.G.add_node("t", type="sink")  # 汇点

        # 添加供应商节点
        for i, sup in enumerate(suppliers):
            node_id = f"s_{sup['id']}"
            self.G.add_node(node_id, type="supplier", stock=sup.get("stock", 0))

        # 添加用户节点
        for j, user in enumerate(users):
            node_id = f"u_{user['id']}"
            self.G.add_node(node_id, type="user", need=user.get("need", 0))

        # 添加边
        # 源点到供应商的边
        for i, sup in enumerate(suppliers):
            node_id = f"s_{sup['id']}"
            self.G.add_edge("s", node_id, capacity=sup.get("stock", 0), cost=0, flow=sup.get("stock", 0))

        # 用户到汇点的边
        for j, user in enumerate(users):
            node_id = f"u_{user['id']}"
            self.G.add_edge(node_id, "t", capacity=user.get("need", 0), cost=0, flow=user.get("need", 0))

        # 供应商到用户的边
        for user in users:
            u_node = f"u_{user['id']}"
            for sid, score in user.get("supplier_scores", []):
                s_node = f"s_{sid}"
                if s_node in self.G.nodes and u_node in self.G.nodes:
                    self.G.add_edge(s_node, u_node, capacity=10 ** 9, cost=-int(score), flow=0)

        # 运行MCMF算法获取流量
        self._run_mcmf()

        # 设置节点位置
        self._set_positions(S, U)

        # 设置节点颜色
        self._set_node_colors()

        # 设置边标签
        if self.show_flow:
            self._set_edge_labels()

    def _run_mcmf(self):
        """运行MCMF算法获取流量分配"""
        # 使用您的MCMF代码
        suppliers = self.inst.get("suppliers", [])
        users = self.inst.get("users", [])
        S = len(suppliers)
        U = len(users)
        n = 1 + S + U + 1
        s = 0
        t = n - 1

        mcmf = MinCostMaxFlow(n)
        supplier_id_to_index = {}

        for i, sup in enumerate(suppliers):
            node = 1 + i
            supplier_id_to_index[sup["id"]] = node
            mcmf.add_edge(s, node, int(sup.get("stock", 0)), 0)

        user_id_to_index = {}
        for j, user in enumerate(users):
            node = 1 + S + j
            user_id_to_index[user["id"]] = node
            mcmf.add_edge(node, t, int(user.get("need", 0)), 0)

        for user in users:
            u_node = user_id_to_index[user["id"]]
            for sid, score in user.get("supplier_scores", []):
                if sid in supplier_id_to_index:
                    sup_node = supplier_id_to_index[sid]
                    mcmf.add_edge(sup_node, u_node, 10 ** 9, -int(score))

        mcmf.init_potential(s)
        total_flow, total_cost = mcmf.solve(s, t)

        # 更新图中的流量
        for sup in suppliers:
            s_idx = supplier_id_to_index[sup["id"]]
            s_node = f"s_{sup['id']}"

            for v, cap, cost, rev in mcmf.graph[s_idx]:
                if 1 + S <= v <= S + U:
                    allocated = mcmf.graph[v][rev][1]
                    if allocated > 0:
                        user_id = users[v - 1 - S]["id"]
                        u_node = f"u_{user_id}"

                        # 更新供应商到用户的边流量
                        if self.G.has_edge(s_node, u_node):
                            self.G[s_node][u_node]["flow"] = allocated

        # 更新源点到供应商的边流量（减去未使用的容量）
        for sup in suppliers:
            s_idx = supplier_id_to_index[sup["id"]]
            s_node = f"s_{sup['id']}"

            # 计算该供应商实际流出的流量
            outflow = 0
            for v, cap, cost, rev in mcmf.graph[s_idx]:
                if 1 + S <= v <= S + U:
                    allocated = mcmf.graph[v][rev][1]
                    outflow += allocated

            # 更新源点到供应商的边流量
            if self.G.has_edge("s", s_node):
                self.G["s"][s_node]["flow"] = outflow

        # 更新用户到汇点的边流量
        for user in users:
            u_idx = user_id_to_index[user["id"]]
            u_node = f"u_{user['id']}"

            # 计算该用户实际接收的流量
            inflow = 0
            for v, cap, cost, rev in mcmf.graph[u_idx]:
                if v == t:
                    allocated = mcmf.graph[v][rev][1]
                    inflow = allocated
                    break

            # 更新用户到汇点的边流量
            if self.G.has_edge(u_node, "t"):
                self.G[u_node]["t"]["flow"] = inflow

    def _node_colors_by_type_with_legend(self):
        # 定义类型到颜色的映射
        color_map = {
            "source": "#BBD5D4",  # red
            "sink": "#D7EAEC",  # blue
            "supplier": "#EFE9D3",  # green
            "user": "#BFC5D5",  # orange
            None: "#bbbbbb"
        }
        node_colors = []
        # 构建 legend handles
        handles = {}
        for n, attr in self.G.nodes(data=True):
            typ = attr.get("type")
            c = color_map.get(typ, color_map[None])
            node_colors.append(c)
            if typ not in handles:
                handles[typ] = mpatches.Patch(color=c, label=str(typ))

        # 在 draw() 里调用：plt.legend(handles=list(handles.values()), loc="upper left")
        return node_colors, list(handles.values())

    def _set_positions(self, S, U):
        """设置节点位置"""
        # 源点在左边
        self.pos["s"] = (0, 0.5)

        # 汇点在右边
        self.pos["t"] = (3, 0.5)

        # 供应商节点在中间左侧
        supplier_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get("type") == "supplier"]
        for i, node in enumerate(supplier_nodes):
            y_pos = (i + 1) / (len(supplier_nodes) + 1)
            self.pos[node] = (1, y_pos)

        # 用户节点在中间右侧
        user_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get("type") == "user"]
        for i, node in enumerate(user_nodes):
            y_pos = (i + 1) / (len(user_nodes) + 1)
            self.pos[node] = (2, y_pos)

    def _set_node_colors(self):
        """设置节点颜色"""
        for node, attr in self.G.nodes(data=True):
            if attr.get("type") == "source":
                self.node_colors.append("red")
            elif attr.get("type") == "sink":
                self.node_colors.append("gray")
            elif attr.get("type") == "supplier":
                self.node_colors.append("green")
            elif attr.get("type") == "user":
                self.node_colors.append("orange")
            else:
                self.node_colors.append("gray")

    def _set_edge_labels(self):
        """设置边标签"""
        for u, v, attr in self.G.edges(data=True):
            if attr.get("flow", 0) > 0:
                if u == "s" or v == "t":
                    # 源点到供应商或用户到汇点
                    self.edge_labels[(u, v)] = f"{attr['flow']}/{attr['capacity']}"
                else:
                    # 供应商到用户
                    self.edge_labels[(u, v)] = f"{attr['flow']} (cost:{-attr['cost']})"

    def draw(self, output_file=None):
        """绘制网络流图"""
        plt.figure(figsize=(12, 8))

        # 绘制节点
        node_colors, legend_handles = self._node_colors_by_type_with_legend()
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors, node_size=2000)
        plt.legend(handles=legend_handles, title="Node type", loc="upper left")

        # 绘制边
        edge_widths = [2 + 0.5 * self.G[u][v].get("flow", 0) / max(1, self.G[u][v].get("capacity", 1))
                       for u, v in self.G.edges()]
        nx.draw_networkx_edges(self.G, self.pos, width=edge_widths, arrows=True, arrowsize=20)

        # 绘制节点标签
        node_labels = {}
        for node, attr in self.G.nodes(data=True):
            if attr.get("type") == "source":
                node_labels[node] = "Source"
            elif attr.get("type") == "sink":
                node_labels[node] = "Sink"
            elif attr.get("type") == "supplier":
                node_labels[node] = f"Supplier {node.split('_')[1]}\nStock: {attr.get('stock', 0)}"
            elif attr.get("type") == "user":
                node_labels[node] = f"User {node.split('_')[1]}\nNeed: {attr.get('need', 0)}"

        nx.draw_networkx_labels(self.G, self.pos, labels=node_labels, font_size=8)

        # 绘制边标签
        if self.show_flow:
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=self.edge_labels, font_size=8)

        plt.title("MCMF Network Flow")
        plt.axis("off")
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"图形已保存到 {output_file}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="MCMF网络流可视化工具")
    parser.add_argument("--inst", required=True, help="实例JSON文件路径")
    parser.add_argument("--out", default=None, help="输出图像文件路径（可选）")
    parser.add_argument("--max-nodes", type=int, default=20, help="最大显示节点数量（默认：20）")
    parser.add_argument("--hide-flow", action="store_true", help="隐藏流量显示")

    args = parser.parse_args()

    # 加载实例数据
    with open(args.inst, "r", encoding="utf-8") as f:
        inst = json.load(f)

    # 创建可视化器
    visualizer = MCMFVisualizer(inst, max_nodes=args.max_nodes, show_flow=not args.hide_flow)
    visualizer.create_graph()
    visualizer.draw(output_file=args.out)


if __name__ == "__main__":
    main()