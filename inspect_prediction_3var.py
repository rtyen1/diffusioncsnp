import h5py
import json
import numpy as np
import torch as th

from ml2_meta_causal_discovery.models.causaltransformernp import CausalProbabilisticDecoder


# ===== 1. 路径 =====
work_dir = "/home/rtyen/projects/CausalStructureNeuralProcess-main/ml2_meta_causal_discovery"
run_name = "gp_3var_prob_rerun_100k"

data_path = f"{work_dir}/datasets/data/synth_training_data/manual_3var_x_to_y_to_z/test/manual_3var_x_to_y_to_z_0.hdf5"
model_dir = f"{work_dir}/experiments/causal_classification/models/{run_name}"
config_path = f"{model_dir}/config.json"
model_path = f"{model_dir}/model_8.pt"

device = "cuda" if th.cuda.is_available() else "cpu"


# ===== 2. 读取 config =====
with open(config_path, "r") as f:
    config = json.load(f)

print("Loaded config:")
print(json.dumps(config, indent=2))


# ===== 3. 构建模型 =====
model = CausalProbabilisticDecoder(
    d_model=config["d_model"],
    emb_depth=1,
    dim_feedforward=config["dim_feedforward"],
    nhead=config["nhead"],
    dropout=0.0,
    num_layers_encoder=config["num_layers_encoder"],
    num_layers_decoder=config["num_layers_decoder"],
    num_nodes=config["num_nodes"],
    n_perm_samples=config["n_perm_samples"],
    sinkhorn_iter=config["sinkhorn_iter"],
    use_positional_encoding=config["use_positional_encoding"],
    device=device,
    dtype=th.float32,
).to(device)

state_dict = th.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

print(f"\nLoaded model from: {model_path}")


# ===== 4. 读取一个 hdf5 样本 =====
with h5py.File(data_path, "r") as f:
    data_idx = 1
    data = f["data"][data_idx]
    label = f["label"][data_idx]

print("\nOriginal data shape:", data.shape)
print("True graph (label):")
print(label)

target_xyz = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 0]])
print("Is true graph X -> Y and Z -> Y ?", np.array_equal(label, target_xyz))

# 和训练时一样做标准化
data = (data - data.mean(axis=0, keepdims=True)) / (data.std(axis=0, keepdims=True) + 1e-8)

# shape -> (batch=1, samples=1000, vars=3)
inputs = th.tensor(data, dtype=th.float32).unsqueeze(0).to(device)


# ===== 5. 先看概率矩阵，再采样很多个图 =====
num_samples = 100
model.n_perm_samples = num_samples

with th.no_grad():
    probs = model.forward(inputs, graph=None, is_training=False, mask=None)
    existence_dist = th.distributions.Bernoulli(probs=probs)
    samples = existence_dist.sample()

# probs shape: (num_samples, batch, num_nodes, num_nodes)
probs_np = probs.detach().cpu().numpy()
samples_np = samples.detach().cpu().numpy()

# 取 batch=0 这个数据集
prob_graphs = probs_np[:, 0]       # shape: (num_samples, 3, 3)
sample_graphs = samples_np[:, 0]   # shape: (num_samples, 3, 3)

print("\nProbability graphs shape:", prob_graphs.shape)
print("First 5 probability matrices:")
for i in range(min(5, num_samples)):
    print(f"\nProb matrix {i}:")
    print(prob_graphs[i])

print("\nSampled graphs shape:", sample_graphs.shape)
print("First 5 sampled graphs:")
for i in range(min(5, num_samples)):
    print(f"\nSample {i}:")
    print(sample_graphs[i])

# ===== 6. 看平均概率矩阵 + 平均采样图 =====
mean_prob_graph = prob_graphs.mean(axis=0)
print("\nMean probability matrix:")
print(mean_prob_graph)

mean_sample_graph = sample_graphs.mean(axis=0)
print("\nMean sampled adjacency (edge frequency):")
print(mean_sample_graph)

# 用平均概率矩阵阈值化成一个代表图
threshold = 0.5
representative_graph = (mean_prob_graph > threshold).astype(int)
np.fill_diagonal(representative_graph, 0)

print(f"\nRepresentative graph from mean probability matrix (threshold={threshold}):")
print(representative_graph)


# ===== 7. 统计 3 变量场景下几种典型图出现频率 =====
empty_graph = np.array([[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]])

g_xy_zy = np.array([[0, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0]])

g_yx_yz = np.array([[0, 0, 0],
                    [1, 0, 1],
                    [0, 0, 0]])

g_xy_only = np.array([[0, 1, 0],
                      [0, 0, 0],
                      [0, 0, 0]])

g_zy_only = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 0]])
g_xy_yz_only = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
g_yx_zy_only = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])

def count_match(graphs, target):
    return sum(np.array_equal(g, target) for g in graphs)

n_empty = count_match(sample_graphs, empty_graph)
n_xy_zy = count_match(sample_graphs, g_xy_zy)
n_yx_yz = count_match(sample_graphs, g_yx_yz)
n_xy_only = count_match(sample_graphs, g_xy_only)
n_zy_only = count_match(sample_graphs, g_zy_only)
n_xy_yz = count_match(sample_graphs, g_xy_yz_only)
n_yx_zy = count_match(sample_graphs, g_yx_zy_only)


print("\nCounts among sampled graphs:")
print("empty graph count        :", n_empty)
print("X -> Y, Z -> Y count     :", n_xy_zy)
print("Y -> X, Y -> Z count     :", n_yx_yz)
print("X -> Y only count        :", n_xy_only)
print("Z -> Y only count        :", n_zy_only)
print("X -> Y, Y -> Z count     :", n_xy_yz)
print("Z -> Y, Y -> X count     :", n_yx_zy)
print("other count              :", num_samples - n_empty - n_xy_zy - n_yx_yz - n_xy_only - n_zy_only -n_yx_zy- n_xy_yz)