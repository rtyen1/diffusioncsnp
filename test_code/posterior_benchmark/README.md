# 三节点 25-DAG 后验实验：独立训练，2000 观测动态子采样

当前入口版本是 `independent-v3`。一次命令只训练一个模型：`diffusion`、`pl` 或
`categorical`。三个模型独立生成数据、独立抽样、独立保存和续训，不再共享训练批次
或抽样计划。它们仍使用相同的 SEM 先验和训练预算。**数据生成与训练均不标准化。**
训练期间不自动验证、不自动测试，也不预先生成测试集；测试是之后的手动操作。
旧 posterior v1/v2 运行目录、旧训练入口和其他 benchmark 结果不会被覆盖。

## 1. 生成分布与参考后验

`all_25_sems.tex` 是原始 SEM 表的项目内快照（仅规范化文本换行），包括三节点
A/B/C 上全部 25 张 DAG、11 个 MEC、固定结构系数和独立高斯外生噪声方差。
图先验是每张 DAG 1/25，不是每个 MEC 1/11。每个 shard 的图标签严格均衡并打乱，
观测噪声独立采样；SEM 参数本身不重新采样。不做节点置换增强或 iSCM 标准化。

约定 B[i,j] 是 i -> j 的系数。列向量 SEM 为 x=B^T x+epsilon，
epsilon~N(0,Omega)，因此 F=(I-B^T)^(-1) Omega^(1/2)，Sigma=F F^T。
默认用对应 Sigma 的 Cholesky 因子 L 生成 X=Z L^T，与逐父子节点生成 SEM 的
**联合观测分布**等价。Cholesky 因子不是图标签；label 始终来自原 SEM 的箭头。
`generate_training_data.py --sampling-method sem` 也支持直接沿父子关系采样。

生成器检查全部 25 个图的方向、拓扑序、MEC、协方差和正定性，并检验逐节点 SEM
与矩阵形式的等价性。已知总体均值是零；对原始 X 使用未经去均值的 S=X^T X：

```text
log p(X|G) = -0.5 * [3*n*log(2*pi) + n*log|Sigma_G| + tr(Sigma_G^(-1)*S)]
p(G|X) = softmax_G(log p(X|G) + log(1/25))
```

这是固定 SEM/点质量参数先验下枚举得到的精确后验（float64 精度），不是共轭参数
先验、BIC、MCMC 或拟合参数后代入的近似。同 MEC 内的似然与图后验严格相等：例如
MEC 4 的 DAG #4/#6/#14，大样本时各接近 1/3，而非某一张接近 1。

生成器保存原始 float64 `data[K,n,3]`、`label[K,3,3]`、`class_index`（0..24）、
`graph_id`（1..25）、`mec_id`、`dataset_seed`、`posterior/log_posterior[K,25]`、
`mec_posterior[K,11]`、`scatter`、`sufficient_statistics` 和逐 dataset 的 X SHA256。
posterior 顺序始终对应文档 DAG #1..#25。网络只读取 X 和图标签；充分统计量和
精确后验不进入训练。训练文件中保存的 2000 观测后验也不作为子样本训练标签。

独立数据生成脚本保留旧协议，NumPy+h5py 即可运行；流式训练会自动调用同一生成器，
不需要先执行独立生成命令。训练文件为 `train/n2000/d3_shard_*.hdf5`，每份 dataset
确实有 2000 条原始观测，既未经验去均值，也未经验标准化。

## 2. 与原训练的关系

直接复用原来的 `CausalTNPEncoder`、`BakStyleSkeletonHead`、GPL denoiser、
PL 概率函数、Kahn 排序和 `DiffusionUtils` 的 riffle 加噪实现。采样直接调用
`datautils.shared_batch_observation_indices`，不重写它的概率分布。
这不是把旧 `train_causal_classify.py` 原封不动运行：需要适配三种输出头、关闭输入
标准化/自动评测，并按下面的实验目标训练；旧的 precedence 辅助头和点估计 rerank
不参与此受控后验实验。默认不带 u，不把它冒称为原 hybrid-e5 带 u 权重的直接延续。

共同架构默认 d_model=256、FFN=512、8 heads、4 层 causal encoder，emb_depth=1，
无节点位置编码，dropout=0.1。skeleton decoder 为 2 层，GPL denoiser 为 7 层。
网络内部 LayerNorm 保留，不等于对输入 X 做标准化。

| method | 训练方式 |
|---|---|
| diffusion | 原来的随机合法序 + riffle 前向加噪 + 多步 GPL 反向去噪，另加普通 skeleton BCE |
| pl | skeleton 与单步 PL 构成的精确图概率 NLL：-log q(G_sampled|X) |
| categorical | 固定 A/B/C 顺序拼接 encoder 节点表示，MLP+25类 softmax，图标签 NLL |

diffusion 默认 `--diffusion-loss denoising`：**直接调用原类的**
`_training_losses_per_batch`，使用时间表 0/4/7/11，sample_N=1。
skeleton 也调用原 `_skeleton_loss_per_batch`，仅设 `skeleton_loss_type=bce`，
不是 balanced/mixed-balanced BCE。默认总损失为 0.2*order_loss + 1.0*skeleton_loss。
训练日志分别显示 loss、order_loss、skeleton_loss，不能把它与另外两模型的 NLL
数值当作同一种训练损失比较。

保留显式 `--diffusion-loss graph_nll` 作为另一种实验：通过有限状态枚举直接优化
完整图 NLL，而不是旧去噪代理目标。不同目标必须使用新 run-root，不能混写实验。

如需 u，可用新目录加 `--priority-samples 16`。diffusion 去噪训练使用原 priority
Kahn 排序和新采样的连续 uniform-u；评测时用 checkpoint 中固定的 16 个 u 近似
边缘积分。PL 也接收 u 值/秩并使用固定样本库定义混合分布。这时不是纯单个 PL；
给定 u 的排列/路径求和精确，但对连续 u 的积分是 MC 近似。默认 0 时全部精确枚举。

## 3. 每批抽样与独立流式训练

每个训练 dataset 保存 2000 条观测，每个 minibatch（默认 16 份 dataset）使用
原规则随机决定一个 n，随后无放回抽取 n 个观测行，batch 内共用行索引：

- 50% log-uniform `[20,2001)`，50% integer-uniform `[100,2001)`；n 在 batch 间交错。
- 同一次训练只产生一个模型，不按 n 分别训练多个模型。
- 三个模型的训练 n 和行索引无需一致，不保存跨模型共享抽样计划。
- 抽取前后都不标准化，只将实际输入转换成网络的 float32。

`--data-seed` 是基础 seed。代码用 method 和 training_seed 派生独立训练数据 seed，
写入 run_config；所以三个命令即使都用默认 seed，也生成独立训练观测。
各自使用自己的 NumPy/Torch RNG；完整 checkpoint 保存 RNG 与 batch cursor。

默认每个模型每批 70000 份新数据、10 批，共 70 万份独立训练 dataset。每份只遍历
一次，每批结束不重置模型或优化器。单批原始 X 约 3.36 GB（3.13 GiB），另有标签、
统计量、checkpoint 开销；不是一次把整个 HDF5 载入 RAM。

优化器 AdamW，lr=2e-4、weight_decay=1e-5，3750 个全局训练步线性 warmup 后恒定
学习率，gradient_clip=1。样本量上下界、hybrid 比例与普通 uniform/log_uniform
均有参数可配置。每个 minibatch 的实际 n 会打印，批次 n 直方图会归档。

```text
模型自己的生成批次 k（每份 2000 观测）
  -> 按旧规则逐 batch 随机抽 n 条，训练这个模型
  -> 保存完整 checkpoint、归档生成 seed/数据指纹/抽样轨迹摘要
  -> 提交完成状态
  -> 可选删除该模型本批的数据
  -> 生成该模型下一批新数据，继续训练
```

`--delete-trained-chunks` 只删除本 run 自己拥有、且 checkpoint 完整保存/校验成功
的那一个 chunk，不等待别的模型，不删除旧训练数据或 checkpoint。
Ctrl+C/SIGTERM 会在训练中等当前 batch 完成后保存；中断批次保留。相同命令加
`--resume` 可恢复数据、抽样 RNG、去噪 RNG、优化器和计数，不重复累计已保存训练步。
首次初始化的安全中断也可恢复。所有训练阶段都不自动验证或测试。

## 4. 上传与测试代码

本地上传（当前 SSH 端口 34688）：

```bash
rsync -ah --exclude='__pycache__/' -e 'ssh -p 34688' \
  /home/rtyen/projects/CausalStructureNeuralProcess-main/test_code/posterior_benchmark/ \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/projects/diffusioncsnp/test_code/posterior_benchmark/
```

也可以使用 git add/commit/push 与服务器 pull --ff-only，但不要强行覆盖工作区冲突。
服务器使用已有 csnp Python，无需 conda activate：

```bash
cd /root/autodl-tmp/projects/diffusioncsnp
/root/autodl-tmp/conda-envs/csnp/bin/python -m unittest \
  test_code.posterior_benchmark.test_generate_training_data \
  test_code.posterior_benchmark.test_training
```

包含旧去噪 loss 数值等价、旧抽样函数等价、生成 2000 再抽样、无标准化、独立训练、
中断/追加恢复、只删本模型数据、手动统一评测和精确 KL 等端到端检查。

## 5. 三个模型分别启动

三个命令各自独立。建议先在一个 screen 中跑 diffusion，再分别启动其他方法；
不建议未经显存检查就同时跑三个 GPU 训练。

```bash
cd /root/autodl-tmp/projects/diffusioncsnp
screen -S posterior_diffusion
```

在 screen 内运行 diffusion：

```bash
/root/autodl-tmp/conda-envs/csnp/bin/python -u test_code/posterior_benchmark/run_streaming.py \
  --method diffusion \
  --run-root /root/autodl-tmp/outputs/posterior_benchmark/d3_diffusion_independent_v3 \
  --chunks 10 --train-per-chunk 70000 --delete-trained-chunks --resume
```

PL：

```bash
/root/autodl-tmp/conda-envs/csnp/bin/python -u test_code/posterior_benchmark/run_streaming.py \
  --method pl \
  --run-root /root/autodl-tmp/outputs/posterior_benchmark/d3_pl_independent_v3 \
  --chunks 10 --train-per-chunk 70000 --delete-trained-chunks --resume
```

25 分类：

```bash
/root/autodl-tmp/conda-envs/csnp/bin/python -u test_code/posterior_benchmark/run_streaming.py \
  --method categorical \
  --run-root /root/autodl-tmp/outputs/posterior_benchmark/d3_categorical_independent_v3 \
  --chunks 10 --train-per-chunk 70000 --delete-trained-chunks --resume
```

每个 run 的 `checkpoints/METHOD_seed0/` 保存 `latest.pt` 与
`checkpoint_chunk_00001.pt` 等完整状态，编号对应新数据批次，不是同一批重复 epoch。
`data_protocols/chunk_00001.json` 保存生成设置、指纹和实际 n 直方图。
`run_state.json` 的 `training_complete` 表示训练完成，不表示已经测试。

中断后原命令重跑即可。增加训练量时保持其他设置不变，改为
`--chunks 20 --resume --extend-target`。加 `--dry-run` 只打印协议，不创建或删除文件。
新的 v3 不会接着旧 v1/v2 目录训练；它们的数据组织/目标不同，必须使用新目录。

## 6. 之后手动评测

训练不共享数据，但比较效果时仍应使用相同测试 X。统一评测入口先固定 checkpoint，
再让全部模型读取同一套固定测试数据。默认测试 n=20/50/100/300/500/1000/2000，
每种 n=5000 份；仅第一次手动测试时才生成，使用独立 evaluation_seed。
测试数据不标准化，精确后验由实际这 n 条原始观测计算，不使用训练文件的后验。

以三个模型都完成第 10 批为例：

```bash
/root/autodl-tmp/conda-envs/csnp/bin/python -u test_code/posterior_benchmark/evaluate.py \
  --run-roots \
    /root/autodl-tmp/outputs/posterior_benchmark/d3_diffusion_independent_v3 \
    /root/autodl-tmp/outputs/posterior_benchmark/d3_pl_independent_v3 \
    /root/autodl-tmp/outputs/posterior_benchmark/d3_categorical_independent_v3 \
  --checkpoint chunk_00010 \
  --output-root /root/autodl-tmp/outputs/posterior_benchmark/d3_compare_independent_v3/chunk_00010
```

单独测试某模型可用 `--run-root`。`last_completed`（默认）和 `latest` 别名均会
先解析成已完成的不可变 chunk archive，**不会评分正在更新的 latest.pt**。
追加训练后评分新的 chunk 会使用新目录，不覆盖之前结果。缓存核验 checkpoint、
源代码、输入数据和输出文件的 hash，不悄悄复用不兼容结果。

图概率统一为 q(G|X)=q(S_G|X) sum_{pi in Topo(G)} q(pi|X)。不阈值化 skeleton，
不选众数/medoid，不额外乘 1/25，也不除以合法序数量。PL 直接算六个排列概率；
diffusion 从均匀终端排列开始，用每步 6x6 转移矩阵在 log 空间求和全部路径。
评测阶段的精确枚举不改变默认去噪训练目标；原独立 skeleton/order 的表达限制保留。

结果目录：

```text
OUTPUT/
  raw_metrics.csv
  summary_by_samples.csv
  summary_across_training_seeds.csv
  evaluation_config.json
  METHOD_seed0/predictions/n100.npz ...
```

NPZ 包含真实与预测的完整 25 维 posterior/log_posterior、图/MEC 编号、dataset_seed
和原始 X SHA256。汇总前检查测试 dataset 一一配对，不要求训练数据或抽样轨迹相同。
主指标为自然对数 KL(p_exact || q_model)，另有 TV、JS、图 NLL、MEC KL、真实 MEC
概率质量和类内条件 TV。`summary_by_samples.csv` 的 mean/std 在每个 model/seed/n
组的单个 dataset 之间计算，SD 用 ddof=1；跨训练 seed 的 SD 单独保存，不混用。
