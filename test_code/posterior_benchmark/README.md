# 三节点、25 张 DAG 的固定先验后验拟合数据

生成入口：`generate_training_data.py`。只需要 NumPy、h5py，不导入模型、PyTorch、
TensorFlow、GPflow，不需要 GPU。模型训练代码不包含在这个数据生成入口内。

## 1. 唯一的生成分布

`all_25_sems.tex` 是用户原始 SEM 表的项目内快照，仅规范化 CRLF -> LF 和文件末尾
换行，文本内容不变。保留原 A/B/C 标签、DAG #1..#25、MEC #1..#11、系数和外生噪声方差。
参数不重新采样。图的先验是每张 DAG 1/25，不是每个 MEC 1/11。

每份数据选择一张 DAG，生成 n 个独立观测，已知总体均值为零。数据不去均值，
不标准化，不使用 iSCM，不做节点置换增广。各 split/n/shard 的图标签严格均衡并
打乱顺序（分层抽样，非跨数据集的独立二项计数），因此单份数据的边缘图先验
仍为 1/25。观测噪声在各份数据及 train/val/test 之间使用独立 seed 流。
三个 split 使用同一 SEM 先验，但不共享观测；这是同先验后验拟合，不是未知图泛化。

## 2. 为什么与父节点生成子节点等价

约定 B[i,j] 是 i -> j 的系数，列向量方程是 x=B^T x+epsilon，
epsilon~N(0,Omega)，Omega 为对角噪声方差矩阵。因此

```text
F = (I - B^T)^(-1) Omega^(1/2)
Sigma = F F^T
```

默认以各 MEC 的 Sigma 做一次 Cholesky 分解 Sigma=L L^T，再以 X=Z L^T 采样，
Z 的各项是独立标准高斯。这与原 SEM **观测分布**相同；不会拿 Cholesky 的下三角
结构代替原 DAG。图标签始终取自原结构方程的非零系数。这里只生成观测数据，
不声称用 Cholesky 因子生成的干预分布也与原 SEM 相同。

启动时自动检查：

- 恰好枚举三节点全部 25 张不同 DAG；方向、拓扑序和 SEM 文本标题一致。
- 标注的 MEC 与 skeleton/v-structure 分组一致。
- 从 25 个 SEM 推出的 Sigma 各自等于文本中的声明；11 个 Sigma 正定。
- 使用同一标准外生噪声时，逐节点 SEM 与矩阵 SEM 生成值一致到浮点精度。
- F 与 L 之间的噪声旋转为正交矩阵；正确耦合噪声后，两种采样值一致。

注意：不能要求给 F 和 L 直接输入同一 Z 就逐元素相同，不同协方差因子通常
对应不同的噪声坐标。相同的是联合观测分布，不是两个任意随机数实现的逐点输出。
若希望直接沿父子关系采样，可使用 `--sampling-method sem`（需要另一个输出目录）。

## 3. 精确后验与充分统计量

对每份原始数据 X，保存未经去均值的 scatter S=X^T X，并计算

```text
log p(X|G) = -1/2 * [n*3*log(2*pi) + n*log|Sigma_G| + tr(Sigma_G^(-1) S)]
p(G|X) = softmax_G(log p(X|G) + log(1/25))
```

不是 BIC、MCMC 或参数估计后再代入的近似；这是固定 SEM / 点质量参数先验下，
数值枚举得到的精确图后验（限于 float64 数值精度），并非共轭参数先验实验。
11 个 MEC 共用 11 个似然，同一 MEC 内每张 DAG 的后验严格相同。
使用 log-sum-exp 防止下溢；同时保存 log_posterior，极小尾概率即使在普通
posterior 中下溢到 0，其对数仍保留。数据与后验全部 float64，后验从实际保存的
同一份数据计算，避免先生成后验、再有损转换输入造成不一致。

`sufficient_statistics` 的七维顺序为
`[n,S_AA,S_BB,S_CC,S_AB,S_AC,S_BC]`。真实后验恰为该向量经过线性 logits 和 softmax；
`catalogue.json` 中还保存了对应解析权重和偏置，便于校验分类头或设置解析参考。
不应使用 np.cov 的默认去均值协方差替代 S，因为总体均值在本模型中已知为零。

## 4. 默认规模、文件与种子

观测数：20、100、500。每个观测数分别有：

| split | 数据集数 | 每张 DAG 数量 |
|---|---:|---:|
| train | 50,000 | 2,000 |
| val | 2,500 | 100 |
| test | 5,000 | 200 |

合计 train=150,000，val=7,500，test=15,000，总计 172,500 份。
每个 HDF5 默认最多 5,000 份。预计原始观测约 856 MB，加上后验、统计量及其他
字段，总体约 1 GB，实际以磁盘文件为准。分批生成、分批写盘，不把整套数据放入内存。

```text
OUTPUT/
  all_25_sems.tex
  catalogue.json
  generation_config.json
  validation_report.json
  train/
    n20/d3_shard_00000.hdf5 ...
    n100/d3_shard_00000.hdf5 ...
    n500/d3_shard_00000.hdf5 ...
  val/n20/...  val/n100/...  val/n500/...
  test/n20/... test/n100/... test/n500/...
```

每个 HDF5 中 K 份数据的主要字段：

| 字段 | 形状/含义 |
|---|---|
| data | [K,n,3]，原始 float64 观测，列顺序 A/B/C |
| label | [K,3,3]，int8，label[i,j]=1 表示 i -> j |
| graph_id | [K]，与文档一致的 1..25 |
| class_index | [K]，分类头用 0..24，等于 graph_id-1 |
| mec_id | [K]，1..11 |
| posterior / log_posterior | [K,25]，顺序对应 DAG #1..#25 |
| mec_posterior | [K,11]，同 MEC 的图概率求和 |
| scatter | [K,3,3]，未经去均值的 X^T X |
| sufficient_statistics | [K,7]，包含 n 的充分统计量 |
| log_evidence | [K]，包含高斯归一化常数的 log p(X) |
| dataset_seed / metadata | 随机种子、图编号、MEC 和 n |

seed 由 master seed、split code、n、全局 dataset index 派生，标签打乱使用另一条
独立流。batch-size 不影响数据；三模型必须读取同一套文件，不分别重生成。
改变总量、分片大小、seed、SEM 文档、采样方式或依赖版本需要新目录。

分片写入隐藏 `.partial.hdf5`，每批 flush 后才记录完成标记。正常中断后原命令
重跑会验证并跳过已完成的行/文件，只补未完成部分。不覆盖不同协议的旧数据。
完成校验会重新核算后验、统计量、图标签、图先验均衡、seed 唯一性和数据 SHA256。

## 5. 服务器命令

在项目目录运行，不需要重新安装模型环境。使用已有 csnp Python：

```bash
cd /root/autodl-tmp/projects/diffusioncsnp
/root/autodl-tmp/conda-envs/csnp/bin/python -u \
  test_code/posterior_benchmark/generate_training_data.py \
  --output-root /root/autodl-tmp/datasets/synth_training_data/posterior_d3_25dag_uniform_fixedsem_n20_100_500_v1 \
  --sample-sizes 20,100,500 \
  --train-per-size 50000 --val-per-size 2500 --test-per-size 5000 \
  --datasets-per-file 5000 --batch-size 128 \
  --sampling-method covariance --seed 20261107
```

成功时末尾显示 `POSTERIOR DATA COMPLETE: datasets=172500`。
原命令加 `--validate-only` 可重新检查，不重采样。代码默认读取同目录打包的 tex，
无需在服务器上额外找到本地 `/home/rtyen/projects/all_25_sems.tex`。

本地/服务器单元测试：

```bash
python -m unittest test_code.posterior_benchmark.test_generate_training_data
```

## 6. 后续训练不能直接套用旧训练命令

旧训练入口可能会逐变量标准化、随机抽观测行或进行其他增广；这会使这里保存的
后验目标不再对应模型看到的原始数据。后续应使用专门的共享先验后验训练入口：
三模型使用相同数据/充分统计量，并明确接收 n；不擅自标准化，不丢观测。
如改变观测子集，要重新计算后验。若重新标记 A/B/C，还会改变固定参数先验，
不能只置换图标签就继续用本实验的似然。网络内部的排序表示变换与数据增广不同，
内部变换只要正确映射回原标签即可。

先验中的不可识别性是刻意保留的：数据足够确定 MEC 4 时，DAG #4/#6/#14 的
后验接近各 1/3，而不是某一张变成概率 1。不能把生成时抽到的 graph_id 当作
唯一正确的后验软标签；posterior 才是后验拟合的目标。
