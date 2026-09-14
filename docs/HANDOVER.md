# 论文交接文档（完整版）

> **生成时间：** 2026-09-14
> **项目根目录：** `/Users/xiexinni/CMGAN-t60-kan-multitask`
> **用途：** 全量交接。接手者读完本文档即可继续工作，无需重新摸索历史。
> **配套文档：** `docs/PAPER_STATUS.md`（数据细节版）、`AGENTS.md`（写作纪律，agent 必读）

---

## ⚠️ 第一件事：分支状态（最重要，先读）

**论文最新改动在 `exp/gradient-remedy` 分支，不是 `exp/kan-multitask` 主线分支！**

| 分支 | 内容 | 状态 |
|---|---|---|
| `exp/gradient-remedy` | **论文全部最新改动**（§5.2 新数据、摘要修复、新图）+ GR 实验代码 | 最新 commit `b4ea9ce` |
| `exp/kan-multitask` | 论文主线 + **MLP 头消融的结果文件**（`runs/kan_multitask_norm_w_denoise_mlphead_b005/`） | 落后，commit `c9e26de` |

**关键问题：MLP 消融的 checkpoint 和测试结果只存在于 `kan-multitask` 分支的 git 里**，gradient-remedy 分支的 `runs/` 看不到它。论文 §5.1 引用了它的数字（102.17/64.94）。**定稿前必须把两个分支合并**（或把 MLP 结果目录 cherry-pick 过来），否则换机器 clone gradient-remedy 分支会拿不到这份数据。

其余分支：`exp/single-t60`（单任务基线，已稳定）、`master`（初始版，不用）。

---

## 一、论文基本信息

- **题目方向：** 去噪辅助监督下的多任务盲 T60 估计（英文：Multi-task blind T60 estimation with denoising auxiliary supervision）
- **核心方法：** T60 回归（主）+ 去噪重建（辅助）共享 dual-path time–frequency Conformer backbone，Fourier-KAN 回归头
- **主模型配置：** α=1.0, β=0.05, EMA 归一化损失（"通过参数调优确定"，论文不写试错过程）

### 论文文件

| 文件 | 绝对路径 | 说明 |
|---|---|---|
| 英文主稿 | `/Users/xiexinni/CMGAN-t60-kan-multitask/docs/main.tex` | 投稿版 |
| 中文 review 版 | `/Users/xiexinni/CMGAN-t60-kan-multitask/docs/main_zh.md` | 与英文同步 |
| 中英对照 | `/Users/xiexinni/CMGAN-t60-kan-multitask/docs/main_bilingual.md` | **⚠️ 已过期**（8/10 生成，落后于 8/24 的 main.tex），用户已表态不再需要自动生成，如需审英文直接对 main.tex |
| 写作纪律 | `/Users/xiexinni/CMGAN-t60-kan-multitask/AGENTS.md` | **agent 必读**，换对话也必须遵守 |
| 数据细节 | `/Users/xiexinni/CMGAN-t60-kan-multitask/docs/PAPER_STATUS.md` | 本文档的前身，数据路径更细 |
| 引用待核实 | `/Users/xiexinni/CMGAN-t60-kan-multitask/docs/引用待核实清单.md` | refs.bib 的 TODO 已清零，此清单仅供追溯 |
| 对标论文 | `docs/DAMTL.pdf`、`docs/noiseaware.pdf` | 写方法/实验章节前先参考它们的叙述方式 |

### 当前论文结构

```
§3 Proposed method（5 小节，无任何参数统计）
  3.1 Dense conv encoder / 3.2 Two-stage Conformer block
  3.3 T60 regression head / 3.4 Denoising decoder / 3.5 Loss function
§4 Experiments
  4.1 Dataset / 4.2 Baselines and metrics / 4.3 Model and training setup（末位，无 α/β 具体值）
§5 Results and discussion
  5.1 Ablation（权重调优 5 档 + MLP/KAN 头消融）
  5.2 Representation analysis（CKA + 线性探针，全部 β=0.05 新数据）
  5.3 Results comparison with baselines
  5.4 SNR 影响 / 5.5 T60 影响（4 方法 × 3 场景分桶表）
  5.6 Denoising results
Contributions 4 条：(1)框架无数字 (2)梯度+CKA (3)Fourier-KAN (4)评估验证
```

---

## 二、核心证据链（§5.2 的叙事，所有数字已核对）

**一句话：去噪把特征"换了格式"（非线性组织），Fourier-KAN 是能读这种格式的读出器，二者协同产生增益。**

| 环节 | 证据 | 数值 | 数据来源 |
|---|---|---|---|
| 1. 梯度无冲突 | cos 均值≈0.028，41% 步 cos<0 但中位数仅 -0.06 | 轻微冲突 | 探针日志（Downloads 的 swanlab log） |
| 2. 表征被重塑 | CKA 深层偏离，排序=去噪权重排序 | Encoder 0.95 / TSCB-2：0.51(去噪主导)<0.70(持平)<0.81(T60主导) | `paper_cka_analysis/` |
| 3. 线性可读性**不升** | 线性探针（冻结特征 + Ridge，双池化，5 seeds±SD + 留一协议） | TSCB-2 multistats：β=0.05 为 0.68±0.01 **<** 单任务 0.74±0.01 | `paper_cka_analysis/results/probe_*.csv` |
| 4. 非线性读出兑现增益 | head 消融（同多任务配置） | KAN 92.91ms < MLP 102.17ms | MLP 结果在 **kan-multitask 分支** |
| 5. 端到端 | 主对比 | 92.91 vs 单任务 111.15（-16.4%） | `runs/.../b005/test/` |

**措辞红线：**
- 禁止写"提升线性可解码性"（数据是下降的）
- 禁止写"权重越大精度越好/单调"（β=0.02/0.01 更差，倒 U 形；只能说"明显主导权重最优"）
- 禁止仅凭低 CKA 声称"表征质量更高"（CKA 只说明"变了"）

---

## 三、全部实验数据速查

### β 扫描（RMSE/MAE，四测试集均值）

| β | RMSE | MAE | 备注 |
|---|---|---|---|
| 0.01 | 95.65 | 61.78 | 已跑，**未列入论文表 2** |
| 0.02 | 102.83 | 67.24 | 已跑，**未列入论文表 2** |
| **0.05** | **92.91** | **56.68** | **主模型** |
| 0.1 | 97.79 | 57.95 | |
| 0.3 | 107.10 | 64.28 | |
| 1.0（持平） | 109.90 | 67.16 | |
| 0.1/1.0（T60主导） | 109.54 | 68.15 | |

论文表 2 只列 5 档（0.05/0.1/0.3/1.0/T60主导），0.02/0.01 藏起来（符合"不写试错"纪律）。

### 对比方法（整体 RMSE，表 4 用）

| 方法 | RMSE | 数据绝对路径 |
|---|---|---|
| BERP | 147.20 | `/Users/xiexinni/CMGAN-t60-kan-multitask/papercompare/berp_t60_v7_fixed/test_results/` |
| DAMTL | 139.33 | `/Users/xiexinni/CMGAN-t60-kan-multitask/papercompare/damtl/test/` |
| noiseaware | 133.84 | `/Users/xiexinni/Downloads/noiseaware/noiseaware/fix-v7dataset-test/` |

**⚠️ 注意 JSON 格式三种不一样**：BERP/Proposed 是 dict；DAMTL 是 list of dict（key 为 `snr_db`/`bin`）；noiseaware 的指标在 `t60_metrics` 下、bin 在 `t60_bin_analysis`。聚合脚本要分别适配。

### 其他关键数字

- 单任务基线（`single_t60_kan_v2_mse`）：111.15 / 72.40 / r=0.9484 / R²=0.8931，861,793 参数。**注意**：`mae_clip1_log`(107.70) 更好但论文统一用 `kan_v2_mse`
- MLP 头消融：102.17 / 64.94（**结果在 kan-multitask 分支**）
- 去噪指标（§5.6）：去噪主导 PESQ 3.427 / STOI 0.927 / SI-SDR 20.98
- 总参数 1,405,554；去噪分支 +0.54M

---

## 四、论文结构重组与修改历史（全量，按轮次）

### 第一轮：数据修正
1. 单任务基线澄清（AGENTS.md 误记 → 修正为 kan_v2_mse）
2. 主模型 β=0.1→0.05（全文数字替换：92.91/56.68/0.9625/16.4%）
3. **删除梯度 r 分析**（effective r 公式方向搞反，"幅度主导"站不住，改为只讲 cos）
4. 表征论点修正（删"重定位到非线性子空间"故事，改为 CKA 确定性数据）

### 第二轮：结构重组（用户逐条指示）
5. §3 拆分（Shared backbone → encoder + Conformer block 两节）
6. §3 去参数化（Parameter budget 移到 §5）
7. §4 重组（Baselines+metrics 合并；training setup 移末尾删三档配置）
8. §5 重组（标题 Results and discussion；5.1 Ablation 合并 head 消融；新增 5.4 SNR / 5.5 T60 分桶表）
9. Contributions 重写（4 条新结构）

### 第三轮：措辞修正（用户逐条审出）
10. 引言动机段 vs 贡献段去重（动机只讲 why，贡献讲 what+结果）
11. "配对"→"联合训练"
12. **删"单调提升"**（全 8 处，中英）
13. **"频率预算"→"频率数/不同数量的三角基"**（自造术语全清）
14. 贡献(2)删权重扫描结论
15. **全文删破折号**（——/---，标准陈述句）
16. 贡献(4)删具体指标数字（对齐 DAMTL/noiseaware 的贡献写法）
17. 摘要"权重越大越好"残留修复（中英，最后一次扫描 2026-09-14）

### 第四轮：§5.2 数据全部换 β=0.05 重跑版
18. CKA 重跑（`paper_cka_analysis/run_paper_cka.py`：全样本 + bootstrap CI + 分测试集），图换 `cka_comparison.png`
19. 线性探针重跑（`run_linear_probe.py`：双池化 + 双协议 + 5 seeds），图换新双面板
20. 叙事升级为三环闭环（CKA 改格式 → 探针线性不升 → KAN 读出兑现），中英同步
21. 结论句收紧（"不高于单任务基线"）

---

## 五、写作纪律（AGENTS.md 固化，agent 换对话必须遵守）

1. 图表引用带编号（图 X / 表 X，不写"如下图"）
2. 参考 DAMTL/noiseaware 的叙述方式（尤其 loss、实验配置、方法章）
3. 中英文版同步（main.tex ↔ main_zh.md）
4. §3 只讲"是什么"，§4 讲"怎么训"；权重值/配置档位不进 §3
5. 换主模型后：包装成"权重敏感性消融"，绝不写试错日志
6. 损失归一化动机不列量级数字
7. **不说"单调"**（数据点少 + 倒 U 形）
8. 三份文件同步（bilingual 已豁免，用户 2026-08 宣布不再需要自动生成）
9. **不用破折号**
10. **不自造术语**

---

## 六、GR（Gradient Remedy）实验——已完成、结论为负、不进论文

| 尝试 | 主辅 | Loss | RMSE | 路径（runs/ 下） |
|---|---|---|---|---|
| v0 | 主=T60 | 归一化 | ~250 | `kan_multitask_norm_w_denoise_b005_gr/` |
| attempt1 | 主=去噪 | 归一化 | 238.78 | `..._b005_gr_t60aux/` |
| attempt2 | 主=去噪 | 原始 | 192.18 | `..._b005_gr_t60aux_raw/` |
| 无GR | — | — | **92.91** | `..._b005/` |

**结论：GR 有害。** 本场景的 41% 冲突步是良性的（cos 中位数 -0.06），GR 解决了不存在的问题，反而破坏去噪对表征的重塑。**不写进论文，留作 rebuttal 弹药**（"我们试过 GR，更差，证明冲突是建设性的"）。

代码：`train_utils/gradient_remedy.py` + `train_multitask.py` 的 `use_gradient_remedy`/`gr_use_raw_loss` 开关。

---

## 七、待定事项（需要用户/导师拍板，非阻塞）

| # | 事项 | 背景 | 成本 |
|---|---|---|---|
| 1 | **可解释性实验**（saliency 按 SNR 分桶 + 与衰减包络对齐） | 导师问过"能否从可解释性论证去噪有效"。现有 CKA/探针是表征分析不是可解释性——缺声学语义。方案已设计（`interpret_t60.py`：梯度归因+物理对齐），**脚本未写** | 低（只需前向+对输入求梯度） |
| 2 | **2×2 冻结换头实验** | 把"协同"从推断变实证的最强补充。现有 head 消融是各自端到端训练（有混淆变量）。2×2 可跑在 `paper_cka_analysis/results/probe_features/` 缓存上（multistats 特征恰好=头输入 192 维） | 极低（CPU 分钟级） |
| 3 | 分支合并 | 见第一节的 ⚠️ | git 操作 |

## 八、投稿前机械清单

- [ ] **LaTeX 编译验证**：本地无编译器，新版 main.tex 从未编译过，投稿前必须 `pdflatex` 跑一遍（多轮手工编辑可能有语法残留）
- [ ] 分支合并（MLP 结果回到论文分支）
- [ ] 图分辨率/字号终检（`cka_comparison.png`、`linear_probe.png`、`gradient_cos_r.png` 是 300dpi 生成，OK；`per_bin.png` 较旧）
- [ ] 作者信息/致谢/基金（main.tex 头部目前只有占位）
- [ ] 投稿格式（cas-sc 模板，目标期刊待定）

---

## 九、给接手 agent 的操作备忘

1. **先读 `AGENTS.md` 的写作纪律**，再动任何论文文件。
2. 改 `main.tex` 必同步 `main_zh.md`；bilingual 不用管。
3. 数据引用一律回源核对：本文档第三节的路径。**DAMTL 的 JSON 是 list 不是 dict**。
4. 服务器环境：Python 在 `/mnt/tidal-sh01/usr/chuan/youling/demucs_xxn/bin/python`，GPU 2/3（GPU 0/1 NVLink 异常），数据集 `/mnt/tidal-sh01/usr/chuan/youling/project/dataset/T60_Dataset_v7_4s`，长任务用 `setsid nohup`。
5. `paper_cka_analysis/results/probe_features/` 缓存（四模型 × mean/multistats 特征）是宝贵资产：可解释性实验、2×2、任何新探针都能直接跑在上面，不用再上 GPU。
6. 用户偏好速记：结果导向不写试错；措辞要过"倒U形""破折号""自造术语"三道筛；贡献条目不堆数字；改完必须 git push（用户在服务器 pull）。
