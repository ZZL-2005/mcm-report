
import matplotlib.pyplot as plt
import networkx as nx

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建图
G = nx.DiGraph()

# 添加节点和边 - 简化版思维导图
edges = [
    ("隐变量模型", "EM算法"),
    ("EM算法", "基本原理"),
    ("基本原理", "E步：计算期望"),
    ("基本原理", "M步：参数更新"),
    ("EM算法", "三硬币问题"),
    ("三硬币问题", "参数估计"),
    ("三硬币问题", "收敛分析"),
    ("EM算法", "可视化实验"),
    ("可视化实验", "参数轨迹"),
    ("可视化实验", "等值面分析"),
    ("隐变量模型", "变分自编码器"),
    ("变分自编码器", "编码器"),
    ("变分自编码器", "解码器"),
    ("变分自编码器", "潜空间"),
    ("变分自编码器", "变分推断"),
    ("隐变量模型", "应用前景"),
    ("应用前景", "数据生成"),
    ("应用前景", "特征学习"),
]

G.add_edges_from(edges)

# 绘图
plt.figure(figsize=(14, 8))
pos = nx.spring_layout(G, k=1.0, iterations=50)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2500, 
        font_size=9, font_weight='bold', arrowsize=15, edge_color='gray')

# 显示图形
plt.title("隐变量模型思维导图", fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()
