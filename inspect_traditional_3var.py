import h5py
import numpy as np

# traditional baselines from causal-learn
# pip install causal-learn
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import fisherz, kci, chisq, gsq


# ===== 1. 路径 =====
work_dir = "/home/rtyen/projects/CausalStructureNeuralProcess-main/ml2_meta_causal_discovery"
data_path = f"{work_dir}/datasets/data/synth_training_data/manual_3var_xzy_to_y/test/manual_3var_xzy_to_y_0.hdf5"
#manual_3var_x_to_y_to_z/test/manual_3var_x_to_y_to_z_0.hdf5
data_path = '/home/rtyen/projects/avici-main/benchmark_data/3/graph_17__0to1__1to2/linear/gaussian/n_100/seed_42/benchmark_3var_numdatasets_10.h5'
# 选哪个样本
data_idx = 1

# ===== 2. 传统方法参数 =====
run_pc = True
run_ges = True

# PC 参数
alpha = 0.05          # 显著性水平，可试 0.05 / 0.01 / 0.001
pc_test = "kci"   # 可选: fisherz, kci, chisq, gsq
pc_stable = True
uc_rule = 0
uc_priority = 2

# GES 参数
# 对连续数据，一般用 local_score_BIC
# 也可以试 local_score_cv_general
sges_score_func = "local_score_BIC"

# 是否先标准化
standardize = True


# ===== 3. 一些工具函数 =====
def maybe_standardize(x: np.ndarray) -> np.ndarray:
    if not standardize:
        return x.astype(np.float64)
    x = x.astype(np.float64)
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)


def cpdag_to_adjacency(cg) -> np.ndarray:
    """
    causal-learn 的 PC 结果 cg.G.graph 是一个邻接矩阵，编码方式通常是：
      graph[i, j] = -1, graph[j, i] =  1   表示 i -> j
      graph[i, j] = -1, graph[j, i] = -1   表示 i - j（无向边）
    这里转成更直观的 [d, d] 矩阵：
      directed[i, j] = 1 表示 i -> j
      undirected[i, j] = undirected[j, i] = 1 表示无向边 i - j
    """
    mat = np.asarray(cg.G.graph).copy()
    d = mat.shape[0]
    directed = np.zeros((d, d), dtype=np.int32)
    undirected = np.zeros((d, d), dtype=np.int32)

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if mat[i, j] == -1 and mat[j, i] == 1:
                directed[i, j] = 1
            elif mat[i, j] == -1 and mat[j, i] == -1:
                undirected[i, j] = 1

    return directed, undirected, mat


def edge_stats_no_diag(true_graph: np.ndarray, pred_graph: np.ndarray):
    d = true_graph.shape[0]
    mask = ~np.eye(d, dtype=bool)

    y_true = true_graph[mask].astype(int).reshape(-1)
    y_pred = pred_graph[mask].astype(int).reshape(-1)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def shd_pairwise(true_graph: np.ndarray, pred_graph: np.ndarray) -> int:
    """
    一个比较直观的 SHD：
    - 漏边、错方向、多边都记错误
    - 对每对节点 {i,j} 只算一次
    适合 3 节点小图做快速比较
    """
    g = true_graph.copy().astype(int)
    h = pred_graph.copy().astype(int)
    np.fill_diagonal(g, 0)
    np.fill_diagonal(h, 0)

    abs_diff = np.abs(g - h)
    mistakes = abs_diff + abs_diff.T
    mistakes_adj = np.where(mistakes > 1, 1, mistakes)
    return int(np.triu(mistakes_adj, k=1).sum())


def print_graph_summary(name: str, true_graph: np.ndarray, pred_graph: np.ndarray):
    cls = edge_stats_no_diag(true_graph, pred_graph)
    shd_val = shd_pairwise(true_graph, pred_graph)

    print(f"\n{name} directed adjacency:")
    print(pred_graph)
    print(f"\n{name} metrics:")
    print(f"SHD:       {shd_val}")
    print(f"TP:        {cls['tp']}")
    print(f"FP:        {cls['fp']}")
    print(f"FN:        {cls['fn']}")
    print(f"Precision: {cls['precision']:.4f}")
    print(f"Recall:    {cls['recall']:.4f}")
    print(f"F1:        {cls['f1']:.4f}")


# ===== 4. 读一个固定测试样本 =====
with h5py.File(data_path, "r") as f:
    data = f["data"][data_idx]     # [n, 3]
    label = f["label"][data_idx]   # [3, 3]

x = maybe_standardize(np.asarray(data))
label = np.asarray(label, dtype=np.int32)

print("=" * 80)
print(f"Data file: {data_path}")
print(f"Chosen sample index inside this hdf5: {data_idx}")
print(f"Data shape: {x.shape}")
print("=" * 80)

print("\nTrue graph (label):")
print(label)

# 你的固定目标图 X -> Y, Z -> Y
# 默认变量顺序就是 0,1,2；通常可理解成 X,Y,Z
print("\nTarget edges in this dataset: 0 -> 1 and 2 -> 1")

print("\nObserved data matrix (first 10 rows):")
print(np.round(x[:10], 4))


# ===== 5. 跑 PC =====
if run_pc:
    print("\n" + "-" * 80)
    print(f"Running PC: alpha={alpha}, indep_test={pc_test}, stable={pc_stable}")

    if pc_test == "fisherz":
        indep_test = fisherz
    elif pc_test == "kci":
        indep_test = kci
    elif pc_test == "chisq":
        indep_test = chisq
    elif pc_test == "gsq":
        indep_test = gsq
    else:
        raise ValueError(f"Unsupported pc_test: {pc_test}")

    cg = pc(
        x,
        alpha=alpha,
        indep_test=indep_test,
        stable=pc_stable,
        uc_rule=uc_rule,
        uc_priority=uc_priority,
        show_progress=False,
    )

    pc_directed, pc_undirected, pc_raw = cpdag_to_adjacency(cg)

    print("\nPC raw graph encoding:")
    print(pc_raw)

    print("\nPC directed edges only:")
    print(pc_directed)

    print("\nPC undirected part of CPDAG:")
    print(pc_undirected)

    print_graph_summary("PC", label, pc_directed)

    print("\nPC key edge status:")
    print(f"0 -> 1 recovered? {bool(pc_directed[0, 1])}")
    print(f"2 -> 1 recovered? {bool(pc_directed[2, 1])}")
    print(f"1 -> 0 wrong dir? {bool(pc_directed[1, 0])}")
    print(f"1 -> 2 wrong dir? {bool(pc_directed[1, 2])}")
    print(f"0 - 1 undirected? {bool(pc_undirected[0, 1])}")
    print(f"1 - 2 undirected? {bool(pc_undirected[1, 2])}")


# ===== 6. 跑 GES =====
if run_ges:
    print("\n" + "-" * 80)
    print(f"Running GES: score_func={sges_score_func}")

    ges_result = ges(x, score_func=sges_score_func)

    # Record['G'] 是最终 CPDAG/graph object，最方便的是看 GraphUtils 不过这里直接取 graph 编码
    ges_graph = ges_result['G']
    ges_raw = np.asarray(ges_graph.graph).copy()

    # 编码规则与 PC 一致处理
    class DummyCG:
        pass
    dummy = DummyCG()
    dummy.G = ges_graph
    ges_directed, ges_undirected, _ = cpdag_to_adjacency(dummy)

    print("\nGES raw graph encoding:")
    print(ges_raw)

    print("\nGES directed edges only:")
    print(ges_directed)

    print("\nGES undirected part of CPDAG:")
    print(ges_undirected)

    print_graph_summary("GES", label, ges_directed)

    print("\nGES key edge status:")
    print(f"0 -> 1 recovered? {bool(ges_directed[0, 1])}")
    print(f"2 -> 1 recovered? {bool(ges_directed[2, 1])}")
    print(f"1 -> 0 wrong dir? {bool(ges_directed[1, 0])}")
    print(f"1 -> 2 wrong dir? {bool(ges_directed[1, 2])}")
    print(f"0 - 1 undirected? {bool(ges_undirected[0, 1])}")
    print(f"1 - 2 undirected? {bool(ges_undirected[1, 2])}")

print("\n" + "=" * 80)
print("Done.")
