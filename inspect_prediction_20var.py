import json
from pathlib import Path

import h5py
import numpy as np
import torch as th

from ml2_meta_causal_discovery.models.causaltransformernp import CausalProbabilisticDecoder


# ===== 1. 路径与参数 =====
work_dir = Path("/home/rtyen/projects/CausalStructureNeuralProcess-main/ml2_meta_causal_discovery")

# 这个数据集名字和仓库里现成 20 节点 probabilistic 模型是匹配的
# 数据目录期望存在于：
#   {work_dir}/datasets/data/synth_training_data/gplvm_neuralnet_20var_ERSFL20U60/test/*.hdf5
# 模型目录期望存在于：
#   {work_dir}/experiments/causal_classification/models/
#      probabilistic_gplvm_neuralnet_20var_ERSFL20U60_NH16_NE4_ND4_DM512_DF2048_BS32/

work_dir = Path("/home/rtyen/projects/CausalStructureNeuralProcess-main/ml2_meta_causal_discovery")

data_file = "gp_20var_fixed_1_2_to_3"
run_name = "probabilistic_gplvm_neuralnet_20var_ERSFL20U60_NH16_NE4_ND4_DM512_DF2048_BS32"
model_filename = "model_2.pt"

split = "test"
file_idx = 0
sample_idx = 0

num_graph_samples = 100           # 采样多少个图
threshold = 0.5                   # 平均概率矩阵阈值化

device = "cuda" if th.cuda.is_available() else "cpu"


def edge_count(g: np.ndarray) -> int:
    return int(g.sum())


def shd(g1: np.ndarray, g2: np.ndarray) -> int:
    return int(np.abs(g1 - g2).sum())


def edge_f1(true_g: np.ndarray, pred_g: np.ndarray):
    true_bin = true_g.astype(bool)
    pred_bin = pred_g.astype(bool)
    tp = int(np.logical_and(true_bin, pred_bin).sum())
    fp = int(np.logical_and(~true_bin, pred_bin).sum())
    fn = int(np.logical_and(true_bin, ~pred_bin).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, precision, recall, f1


def list_top_edges(mean_prob_graph: np.ndarray, k: int = 20):
    n = mean_prob_graph.shape[0]
    items = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            items.append((i, j, float(mean_prob_graph[i, j])))
    items.sort(key=lambda x: x[2], reverse=True)
    return items[:k]


# ===== 2. 拼接路径 =====
data_dir = work_dir / "datasets" / "data" / "synth_training_data" / data_file / split
model_dir = work_dir / "experiments" / "causal_classification" / "models" / run_name
config_path = model_dir / "config.json"
model_path = model_dir / model_filename

if not data_dir.exists():
    raise FileNotFoundError(f"Data directory not found: {data_dir}")
if not model_dir.exists():
    raise FileNotFoundError(f"Model directory not found: {model_dir}")
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

hdf5_files = sorted(data_dir.glob("*.hdf5"))
if len(hdf5_files) == 0:
    raise FileNotFoundError(
        f"No .hdf5 files found under: {data_dir}\n"
        "说明：你当前本地目录里可能只有 graph_args.json，还没有真正的数据文件。"
    )
if not (0 <= file_idx < len(hdf5_files)):
    raise IndexError(f"file_idx={file_idx} 超出范围，当前共有 {len(hdf5_files)} 个 hdf5 文件")

data_path = hdf5_files[file_idx]

print("=" * 80)
print("Using device:", device)
print("Data file:", data_file)
print("Model run:", run_name)
print("Chosen hdf5:", data_path)
print("Chosen sample index inside this hdf5:", sample_idx)
print("=" * 80)


# ===== 3. 读取 config =====
with open(config_path, "r") as f:
    config = json.load(f)

print("\nLoaded config:")
print(json.dumps(config, indent=2))


# ===== 4. 构建模型 =====
model = CausalProbabilisticDecoder(
    d_model=config["d_model"],
    emb_depth=config["emb_depth"],
    dim_feedforward=config["dim_feedforward"],
    nhead=config["nhead"],
    dropout=config["dropout"],
    num_layers_encoder=config["num_layers_encoder"],
    num_layers_decoder=config["num_layers_decoder"],
    num_nodes=config["num_nodes"],
    n_perm_samples=config["n_perm_samples"],
    sinkhorn_iter=config["sinkhorn_iter"],
    use_positional_encoding=config["use_positional_encoding"],
    mlp_use_bias=True,
    device=device,
    dtype=th.float32,
).to(device)

try:
    state_dict = th.load(model_path, map_location=device, weights_only=True)
except TypeError:
    state_dict = th.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print(f"\nLoaded model from: {model_path}")


# ===== 5. 读取一个 hdf5 样本（一个 batch 里的一个数据集） =====
with h5py.File(data_path, "r") as f:
    num_datasets_in_file = f["data"].shape[0]
    if not (0 <= sample_idx < num_datasets_in_file):
        raise IndexError(
            f"sample_idx={sample_idx} 超出范围，这个文件里共有 {num_datasets_in_file} 个数据集"
        )
    data = f["data"][sample_idx]
    label = f["label"][sample_idx]

print("\nOriginal data shape:", data.shape)
print("True graph shape:", label.shape)
print("True graph edge count:", edge_count(label))
print("True graph adjacency matrix:")
print(label.astype(int))

# 和训练时一样做标准化
mean_before = data.mean(axis=0)
std_before = data.std(axis=0)
data = (data - mean_before[None, :]) / (std_before[None, :] + 1e-8)

print("\nPer-variable mean before standardization (first 10):")
print(mean_before[:10])
print("Per-variable std before standardization (first 10):")
print(std_before[:10])
print("Standardized data shape:", data.shape)

# shape -> (batch=1, samples=1000, vars=20)
inputs = th.tensor(data, dtype=th.float32).unsqueeze(0).to(device)


# ===== 6. 先看概率矩阵，再采样很多个图 =====
model.n_perm_samples = num_graph_samples

with th.no_grad():
    probs = model.forward(inputs, graph=None, is_training=False, mask=None)
    existence_dist = th.distributions.Bernoulli(probs=probs)
    samples = existence_dist.sample()

# probs shape: (num_graph_samples, batch, num_nodes, num_nodes)
probs_np = probs.detach().cpu().numpy()
samples_np = samples.detach().cpu().numpy()

# 取 batch=0 这个数据集
prob_graphs = probs_np[:, 0]      # (num_graph_samples, 20, 20)
sample_graphs = samples_np[:, 0]  # (num_graph_samples, 20, 20)

print("\nProbability graphs shape:", prob_graphs.shape)
print("Sampled graphs shape:", sample_graphs.shape)

print("\nFirst probability matrix:")
print(prob_graphs[0])
print("\nFirst sampled graph:")
print(sample_graphs[0].astype(int))


# ===== 7. 平均概率矩阵 + 平均采样图 =====
mean_prob_graph = prob_graphs.mean(axis=0)
mean_sample_graph = sample_graphs.mean(axis=0)

print("\nMean probability matrix:")
print(mean_prob_graph)
print("\nMean sampled adjacency (edge frequency):")
print(mean_sample_graph)

representative_graph = (mean_prob_graph > threshold).astype(int)
np.fill_diagonal(representative_graph, 0)

print(f"\nRepresentative graph from mean probability matrix (threshold={threshold}):")
print(representative_graph)
print("Representative graph edge count:", edge_count(representative_graph))


# ===== 8. 和真图做简单比较 =====
exact_match_flags = [np.array_equal(g.astype(int), label.astype(int)) for g in sample_graphs]
exact_match_count = int(np.sum(exact_match_flags))

sample_shds = [shd(g.astype(int), label.astype(int)) for g in sample_graphs]
rep_shd = shd(representative_graph, label.astype(int))

tp, fp, fn, precision, recall, f1 = edge_f1(label.astype(int), representative_graph)

print("\nComparison against true graph:")
print("Exact true-graph matches among sampled graphs:", exact_match_count, f"/ {num_graph_samples}")
print("Mean SHD of sampled graphs:", float(np.mean(sample_shds)))
print("Min  SHD of sampled graphs:", int(np.min(sample_shds)))
print("Max  SHD of sampled graphs:", int(np.max(sample_shds)))
print("Representative graph SHD:", rep_shd)
print("Representative graph edge-level stats:")
print(f"  TP={tp}, FP={fp}, FN={fn}")
print(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")


# ===== 9. 打印概率最高的若干条边 =====
print("\nTop-20 directed edges by mean probability:")
for rank, (i, j, p) in enumerate(list_top_edges(mean_prob_graph, k=20), start=1):
    print(f"{rank:02d}. {i} -> {j}: {p:.6f} | true_edge={int(label[i, j])}")


# ===== 10. 统计采样图的边数分布 =====
sampled_edge_counts = [edge_count(g) for g in sample_graphs]
vals, cnts = np.unique(sampled_edge_counts, return_counts=True)
print("\nSampled graph edge-count distribution:")
for v, c in zip(vals, cnts):
    print(f"edges={int(v):3d}: count={int(c)}")

print("\nDone.")
