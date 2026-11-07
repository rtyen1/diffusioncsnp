# causalAssembly Station1 六节点重复测试集

生成入口：`test_code/realworldtest/generate_causalassembly_station1.py`。
它是独立的数据准备脚本，不依赖 CSNP 的模型或 PyTorch；请使用 causalassembly
环境运行。运行时调用安装好的官方 `ProcessCell`、`fit_drf`、`sample_from_drf`。

## 协议

- 官方参考数据为公开的半合成观测数据，不是原始工厂数据。
- 从参考图取 Station1：6 节点、7 条边，按 `Station1_mp_0,...,Station1_mp_5` 排列。
- 在 Station1 子图下重新拟合其条件分布，不直接把完整装配线数据的边缘分布当作
  一个无混杂子图的生成分布。
- 默认用全部 15,581 行参考数据，只拟合一次 DRF。`--fit-rows 500` 可以复现
  README 示范的 500 行拟合规模，但应更换输出目录并在报告中注明。
- 沿用官方 `num_trees=2000,min_node_size=15,splitting_rule=FourierMMD`。
- n 为 20/50/100/300/1000/2000，每个 n 独立生成 100 份数据，共 600 份。
  seed 由 `SeedSequence([42,n,replicate,1])` 派生，不是嵌套子集。
- 所有数据共享同一 DAG 和拟合后的条件分布；变化的是新采样的观测。
- 只调用纯观测采样接口；无干预、无额外逐变量标准化；以 float32 存储。
- 每次最多向官方采样器请求 2048 行，避免把全部 100 份数据一次性转成 R 的
  巨型条件权重矩阵。对于 15,581 行拟合数据，单个 2048 行条件权重矩阵约
  244 MiB；转换与森林本身还需要额外内存。可减小 sampling-batch-rows，但应
  保持重跑/恢复时一致。

## 服务器依赖

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /root/autodl-tmp/conda-envs/causalassembly
export RPY2_CFFI_MODE=ABI
cd /root/autodl-tmp/projects/causalAssembly
Rscript --vanilla -e 'options(timeout=300); install.packages("https://cran.r-project.org/src/contrib/drf_1.3.1.tar.gz", repos=NULL, type="source", lib=.libPaths()[1]); stopifnot(requireNamespace("drf", quietly=TRUE)); print(packageVersion("drf"))'
python -m pip install -r requirements.txt h5py -e .
python -m pip check
```

## 生成

确保该 Python 文件已经上传/通过自己的 diffusioncsnp 仓库同步至服务器。
注意它不在 Bosch 官方仓库中。然后运行：

```bash
cd /root/autodl-tmp/projects/diffusioncsnp
python -u test_code/realworldtest/generate_causalassembly_station1.py \
  --source-root /root/autodl-tmp/projects/causalAssembly \
  --output-root /root/autodl-tmp/datasets/benchmark_data_causalassembly_station1_d6_r100_seed42_v1 \
  --sample-sizes 20,50,100,300,1000,2000 \
  --datasets-per-cell 100 \
  --fit-rows 0 \
  --sampling-batch-rows 2048 \
  --seed 42
```

首次 DRF 拟合使用 CPU，可能较慢；它不是在训练 CSNP 或任何待评测方法。
推荐在 screen 中运行。拟合后会保存生成器，之后每完成一份数据都会打印进度。

在同一条命令末尾增加 `--validate-only` 可重新检查所有最终 HDF5，不重新采样。

## 输出与恢复

```text
benchmark_data_causalassembly_station1_d6_r100_seed42_v1/
  generation_config.json
  validation_report.json
  generator/
    station1_drf.pkl
    cache_manifest.json
  causalassembly_station1/
    d6/
      n20/test/d6_shard_00000.hdf5
      n50/test/d6_shard_00000.hdf5
      n100/test/d6_shard_00000.hdf5
      n300/test/d6_shard_00000.hdf5
      n1000/test/d6_shard_00000.hdf5
      n2000/test/d6_shard_00000.hdf5
```

每个 HDF5 有 `data [100,n,6]`、`label [100,6,6]`、`metadata [100]`，列名在
`columns_json`，邻接矩阵的 `[i,j]=1` 表示 i -> j，metadata 记录每份数据的 seed。
可以被现有统一评测的 manifest 发现。后续评测需显式选择节点数 6，并允许部分
benchmark 布局，而不是套用原来 d3/d5/d10/d20 的默认完整布局。

`generation_config.json` 保存参考数据/图/代码校验和、版本、拟合行数、生成参数。
`validation_report.json` 记录完成情况。数据主体约 8.3 MB，另有压缩、metadata
开销；DRF 缓存可能比观测数据文件大。

脚本不会覆盖不同协议的现有结果。若相同配置重跑，会验证并跳过已有最终文件；
同一 cell 的中间进度写入隐藏的 `.partial.hdf5`，每份数据完成后 flush 并记录哈希。
中断时未完成的那一份会重采样，已完成的份数保留。一个 cell 全部完成后才以原子
重命名发布成可评测的最终文件。若中断发生在首次 DRF 拟合完成之前，则需重新拟合。
缓存只允许来自自己在受信任目录生成的文件，不要加载外部来源的 pickle。

标准差应解释为固定一张工业参考图与固定生成机制下，重复采样观测所导致的波动，
不是不同随机图之间的波动。本脚本只生成数据，不运行模型预测或指标评测。
