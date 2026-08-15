# 论文级 CKA 表征分析说明

本目录用于复现 CMGAN-T60 单任务模型与多任务模型之间的共享层表征相似性分析。分析对象包括 Dense Encoder、TSCB-1 和 TSCB-2，主模型为去噪主导配置（α=1，β=0.05）。

## 实验设计

- 数据：test1–test4，每个测试集使用全部 1080 条语音，共 4320 条。
- 对齐：同一测试集内，两个模型始终使用顺序完全一致的输入。
- 表征：将每层 `(C,T,F)` 输出沿时间和频率维做全局平均，得到每条语音的 64 维向量。
- 指标：中心化线性 CKA。实现使用与 Gram 矩阵公式数学等价的特征空间公式，避免构造 `N×N` 矩阵。
- 不确定性：每个“测试集 × 模型 × 层”进行 1000 次配对 bootstrap；每次从样本索引中有放回抽取 N 个索引，并对两模型使用相同索引。95% CI 为 bootstrap 分布的 2.5% 和 97.5% 分位数。
- 跨条件汇总：报告四测试集 CKA 的宏平均与测试集间样本标准差。
- 敏感性检查：额外拼接四测试集的 4320 条表征，计算 pooled CKA；它不是主要结论依据。

## 输出文件

运行完成后，`results/` 包含：

- `cka_results.csv`：逐测试集结果、宏平均/标准差和 pooled CKA。
- `bootstrap_distributions.npz`：每个实验单元的全部 bootstrap 样本。
- `metadata.json`：数据、模型 checkpoint、随机种子和统计方法。
- `features/*.npz`：缓存的逐模型逐测试集特征，可用于复核或断点续跑。
- `figures/fig_primary_b005_per_split_ci95.{png,pdf}`：β=0.05 主模型在四测试集上的逐层 CKA 和 95% CI。
- `figures/fig_all_models_macro_mean_sd.{png,pdf}`：不同损失权重模型的宏平均 CKA 与测试集间标准差。
- `run.log`：完整运行日志。

## 复现命令

```bash
cd /mnt/tidal-sh01/usr/chuan/youling/project/11/multitask-experiments/kan-multitask
setsid nohup /mnt/tidal-sh01/usr/chuan/youling/demucs_xxn/bin/python \
  paper_cka_analysis/run_paper_cka.py \
  --gpu 2 --bootstrap 1000 \
  > paper_cka_analysis/results/run.log 2>&1 < /dev/null &
```

脚本默认复用已存在的特征缓存。只有在 checkpoint 或特征提取逻辑改变后，才应增加 `--force_extract`。

## 结果解释纪律

较低的 CKA 表示多任务表征相对单任务表征发生了更明显的变化，但不等价于“表征质量更高”。论文中应将 CKA 与线性探针和最终 T60 指标共同解释：CKA 描述表征变化，线性探针描述 T60 信息的可线性解码性，RMSE/MAE 描述最终任务性能。

论文正文引用图表时必须使用编号，例如“如图 X 所示”或 `Fig.~\ref{fig:cka_layerwise}`，不能写“如下图所示”。
