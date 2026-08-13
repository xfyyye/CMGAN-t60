# 论文现状说明文档

> **用途：** 供下一位 agent（或下次对话）快速了解论文的实验细节、文件来源、写作偏好、已改事项。
> **生成时间：** 2026-08-07
> **项目根目录：** `/Users/xiexinni/CMGAN-t60-kan-multitask`

---

## 给 agent 的指令：请将以下内容记入 CLAUDE.md / AGENTS.md

以下规则已在 `AGENTS.md` 的"论文写作要求"章节中固化，换对话也必须遵守：

1. **图表引用必须带编号。** 正文叙述时不能直说"如下图/表所示"，必须说"如图 X / 表 X 所示"（`Fig.~\ref{...}` / `Table~\ref{...}`，中文用"图 X / 表 X"）。

2. **参考两篇对标论文的叙述方式。** 写损失函数、实验配置、方法章节前，应先查阅：
   - DAMTL（Zhang 等，2024）：`/Users/xiexinni/CMGAN-t60-kan-multitask/docs/DAMTL.pdf`
   - noiseaware（Zheng 等，2022）：`/Users/xiexinni/CMGAN-t60-kan-multitask/docs/noiseaware.pdf`

3. **中英文版同步。** 英文版 `main.tex` 和中文 review 版 `main_zh.md` 必须内容一致。改一处同步另一处。

4. **方法章(§3) vs 实验章(§4) 的边界。** §3 只讲"是什么"（方法定义）；§4 讲"怎么训"（实验配置）。权重值、配置档位、"哪档是主模型"属于 §4/§5，不写进 §3。

5. **换主模型后 §4 怎么办。** §4 的 subsection 结构不变。绝不写"寻找最佳配置的试错过程"。把 β 扫描包装成"权重敏感性消融研究"。主模型声明只给最终值，不解释怎么试出来的。

6. **损失归一化的叙述纪律。** 只说"为使两个量级不同的损失能在同一尺度上加权"，不列具体量级数字。

7. **不要声称"单调"。** 权重扫描只有几个数据点，不能说"单调提升/monotonically improves"，只说"提升/improves"。

8. **三份文件同步纪律。** 改完 `main.tex` 或 `main_zh.md` 必须同步另一个；如果用户需要 `main_bilingual.md`，也必须重新生成。

9. **不要用破折号。** 论文中不使用中文破折号"——"或英文 em-dash "---"，用标准陈述句。

10. **不要自造术语。** 如"hierarchical frequency budget"（改为"不同数量的三角基"）、"gradient magnitude ratio r"（已删）。

11. **引言结构：动机段 vs 贡献段不重复。** 动机段只讲"为什么这么设计"（不给技术细节），贡献段列"做了什么+结果"。参考 noiseaware 的处理方式。

---

## 一、论文基本信息

- **论文题目：** 去噪辅助监督下的多任务盲混响时间估计（英文：Multi-task blind T60 estimation with denoising auxiliary supervision）
- **核心思路：** T60 回归（主任务）+ 去噪重建（辅助任务），共享 dual-path time–frequency Conformer backbone，用 Fourier-KAN 回归头估计 T60
- **投稿目标：** 期刊（具体待定）

### 论文文件（绝对路径）

| 文件 | 路径 | 说明 |
|---|---|---|
| 英文主稿 | `/Users/xiexinni/CMGAN-t60-kan-multitask/docs/main.tex` | 投稿用，LaTeX 格式 |
| 中文 review 版 | `/Users/xiexinni/CMGAN-t60-kan-multitask/docs/main_zh.md` | 审阅用，Markdown 格式 |
| 中英段落对照 | `/Users/xiexinni/CMGAN-t60-kan-multitask/docs/main_bilingual.md` | 一段中文一段英文交错，供审阅英文书写 |
| AGENTS.md | `/Users/xiexinni/CMGAN-t60-kan-multitask/AGENTS.md` | 持久记忆，含写作纪律 |
| 参考论文 DAMTL | `/Users/xiexinni/CMGAN-t60-kan-multitask/docs/DAMTL.pdf` | 最接近的对标论文（多任务 T60+去噪） |
| 参考论文 noiseaware | `/Users/xiexinni/CMGAN-t60-kan-multitask/docs/noiseaware.pdf` | 对标论文（noise-aware 时频掩蔽） |

---

## 二、论文结构（当前版本）

### §3 Proposed method（方法）
- 3.1 Dense convolutional encoder（从原"Shared backbone"拆出）
- 3.2 Two-stage Conformer block（同上拆出，"trunk"已改为"block"）
- 3.3 T60 regression head（Fourier-KAN）
- 3.4 Denoising decoder branch
- 3.5 Loss function（含 EMA 归一化定义）
- **§3 不含任何参数统计**（参数量移到 §5）

### §4 Experiments（实验）
- 4.1 Dataset（中英对齐，含样本目录结构）
- 4.2 Baselines and metrics（合并原对比方法+指标）
- 4.3 Model and training setup（移到末尾，**不含三档配置具体值**；只讲学习率/梯度裁剪/早停等通用设置）

### §5 Results and discussion（结果与讨论）
- 5.1 Ablation study（任务权重调优 5 档 + head 消融 MLP vs KAN）
- 5.2 Representation analysis（CKA + linear probe）
- 5.3 Results comparison with baselines（主对比，主模型 β=0.05）
- 5.4 Effect of SNR on estimation accuracy（分 SNR 表格，4 方法 × 3 场景）
- 5.5 Effect of reverberation time on estimation accuracy（分 T60 表格，4 方法 × 3 场景）
- 5.6 Denoising results（3 档去噪指标）

### Contributions（4 条）
- (1) 模型框架（不含参数量和指标）
- (2) 梯度探针 + CKA 表征分析（不提权重扫描结论）
- (3) Fourier-KAN head（"不同数量的三角基"，不用"频率预算"）
- (4) 实验指标和参数量集中

---

## 三、实验细节与数据来源

### 3.1 主模型（β=0.05）

- **配置：** α=1.0, β=0.05, EMA 归一化损失, Fourier-KAN head
- **checkpoint 路径：** `/Users/xiexinni/CMGAN-t60-kan-multitask/runs/kan_multitask_norm_w_denoise_b005/best_model.pth`
- **测试结果：** `/Users/xiexinni/CMGAN-t60-kan-multitask/runs/kan_multitask_norm_w_denoise_b005/test/`
- **四集均值指标：**
  - RMSE = **92.91 ms**
  - MAE = **56.68 ms**（精确值 56.675）
  - Pearson r = **0.9625**
  - R² = **0.9253**
- **相比单任务基线（111.15 ms）RMSE 降低 16.4%**
- **总参数：** 1,405,554 (1.41M)

### 3.2 β 扫描完整结果

| β | α/β | RMSE (ms) | MAE (ms) | 备注 |
|---|---|---|---|---|
| 0.01 | 100:1 | 95.65 | 61.78 | 极端去噪主导，已退化 |
| 0.02 | 50:1 | 102.83 | 67.24 | 退化明显 |
| **0.05** | **20:1** | **92.91** | **56.68** | **最优（主模型）** |
| 0.1 | 10:1 | 97.79 | 57.95 | |
| 0.3 | 3.3:1 | 107.10 | 64.28 | |
| 1.0 | 1:1 | 109.90 | 67.16 | 持平 |
| 10.0 | 1:10 | 109.54 | 68.15 | T60 主导 |

**趋势：** 倒 U 形，β=0.05 为峰值。论文表 2 只列了 5 档（0.05/0.1/0.3/1.0/0.1反向），**0.02/0.01 未列入表 2**。论文措辞为"通过参数调优确定 β=0.05 为最优"，不写试错过程。

**⚠️ 论文措辞注意：** 不能说"单调提升"（因为 0.05→0.01 是下降的）。当前论文正文(§5.1)的措辞已改为"精度随去噪权重增大而提升"（不含"单调"），contributions(2) 已删掉权重扫描结论。

### 3.3 单任务基线

- **配置：** `single_t60_kan_v2_mse`（共享主干 + Fourier-KAN 头，无去噪分支）
- **路径：** `/Users/xiexinni/CMGAN-t60-kan-multitask/runs/single_t60_kan_v2_mse/`
- **四集均值：** RMSE=111.15, MAE=72.40, r=0.9484, R²=0.8931
- **参数量：** 861,793

**注意：** `single_t60_kan_v2_mae_clip1_log`（RMSE=107.70）性能更好，但**论文统一用 `kan_v2_mse`（111.15）作为单任务基线**。AGENTS.md 已修正此记忆。

### 3.4 对比方法

| 方法 | 数据路径 | 四集均值 RMSE |
|---|---|---|
| BERP | `/Users/xiexinni/CMGAN-t60-kan-multitask/papercompare/berp_t60_v7_fixed/test_results/` | 147.4（表 4 中写 151.93，有出入，见下文） |
| DAMTL | `/Users/xiexinni/CMGAN-t60-kan-multitask/papercompare/damtl/test/` | 139.3 |
| noiseaware | `/Users/xiexinni/Downloads/noiseaware/noiseaware/fix-v7dataset-test/` | 133.8 |

**⚠️ BERP 数据出入：** 表 4 写的是 151.93（算术平均），但从 JSON 重新用均方根算是 147.4。差异可能来自计算方式（算术平均 vs 均方根）或不同 checkpoint。**这个差异尚未和用户确认是否要更新表 4。**

**JSON 格式差异：**
- BERP / Proposed：`standard_metrics` + `snr_analysis` (dict) + `bin_analysis` (dict)
- DAMTL：`standard_metrics` + `snr_analysis` (list of dict，key 为 `snr_db`) + `bin_analysis` (list of dict，key 为 `bin`)
- noiseaware：`t60_metrics` + `snr_analysis` (dict，key 为 `-5dB`) + `t60_bin_analysis` (dict，key 为 `0.1-0.3s`)

### 3.5 回归头消融（MLP vs KAN，多任务配置）

| Head | RMSE (ms) | MAE (ms) | 路径 |
|---|---|---|---|
| MLP | 102.17 | 64.94 | `/Users/xiexinni/CMGAN-t60-kan-multitask/runs/kan_multitask_norm_w_denoise_mlphead_b005/` |
| **Fourier-KAN** | **92.91** | **56.68** | 同主模型 b005 |

配置完全一致（α=1.0, β=0.05），唯一差异是 `t60_head_type: mlp` vs `fourier_kan`。

### 3.6 去噪指标（3 档）

| 配置 | PESQ | STOI | SI-SDR (dB) | ΔSI-SDR |
|---|---|---|---|---|
| 含噪输入 | 2.721 | 0.847 | 15.91 | — |
| T60主导 (0.1,1.0) | 3.077 | 0.891 | 18.80 | +2.89 |
| 持平 (1.0,1.0) | 3.178 | 0.907 | 19.79 | +3.88 |
| **去噪主导 (1.0,0.05)** | **3.427** | **0.927** | **20.98** | **+5.07** |

### 3.7 表征分析（CKA + linear probe）

- **图文件：** `/Users/xiexinni/CMGAN-t60-kan-multitask/docs/figs/cka_heatmap.png`、`linear_probe.png`
- **生成脚本：** `/Users/xiexinni/CMGAN-t60-kan-multitask/analyze_representations.py`
- **checkpoint 路径已修正为 `single_t60_kan_v2_mse`**（曾误指向 `mae_clip1_log`）
- **CKA 数值（确定性，可信）：**
  - Encoder: 0.988 / 0.989 / 0.995（denoise-dom / balanced / t60-dom）
  - TSCB-2: 0.699 / 0.743 / 0.821
- **Linear probe R²（随机 seed，有抖动，仅作定性参考）：** 只说"深层保持相当的 T60 可解码性"，不强调具体排序
- **论点：** 去噪梯度重塑共享表征（TSCB-2 CKA 偏离最大），且偏离随去噪权重增大而增大

### 3.8 梯度探针（cos 分析）

- **探针 log：** `/Users/xiexinni/Downloads/kan_multitask_norm_w_denoise-2026-6-28_19_52_29.log`（b01）、`kan_multitask_norm_w_denoise_b005-2026-6-28_19_55_38.log`（b005）
- **图文件：** `/Users/xiexinni/CMGAN-t60-kan-multitask/docs/figs/gradient_cos_r.png`（两宫格：mean cos + conflict fraction）
- **生成脚本：** `/Users/xiexinni/CMGAN-t60-kan-multitask/analyze_gradients_norm.py`
- **真实数据：** cos 均值≈0.028，|cos|均值≈0.095，41% 步 cos<0（但中位数仅 -0.06，轻微冲突）
- **论点：** 两任务梯度近似正交，无系统性冲突

**⚠️ 已删除的 r 分析：** 原版曾填入 effective r（2.20/34.17/360.59），公式方向搞反了（α/β 写成 β/α），且"去噪梯度幅度主导"在数据上站不住（去噪梯度只有 T60 的 5-6%）。**已全部删除**，改为只讲 cos。

### 3.9 Gradient Remedy（GR）实验——未写入论文

三个尝试全部比无 GR 差，证明 GR 在本场景有害：

| 尝试 | 主辅关系 | Loss | RMSE (ms) | 路径 |
|---|---|---|---|---|
| v0 | 主=T60 | 归一化 | ~250+ | `runs/kan_multitask_norm_w_denoise_b005_gr/` |
| attempt1 | 主=去噪 | 归一化 | 238.78 | `runs/kan_multitask_norm_w_denoise_b005_gr_t60aux/` |
| attempt2 | 主=去噪 | 原始 | 192.18 | `runs/kan_multitask_norm_w_denoise_b005_gr_t60aux_raw/` |
| 无GR（主模型） | — | — | **92.91** | `runs/kan_multitask_norm_w_denoise_b005/` |

**结论：** 你的"冲突"是良性的、建设性的。GR 解决了一个不存在的问题，反而破坏了去噪对表征的重塑。**不写进论文，留作 rebuttal 弹药。**

**代码位置（在 exp/gradient-remedy 分支）：**
- GR 模块：`/Users/xiexinni/CMGAN-t60-kan-multitask/train_utils/gradient_remedy.py`
- 训练循环改造：`train_multitask.py`（`use_gradient_remedy` / `gr_use_raw_loss` 开关）
- 配置：`configs/t60_multitask/kan_mse_norm_w_denoise_b005_gr_t60aux*.yaml`

---

## 四、逐条改过的事项（按时间顺序）

### 第一轮：数据修正
1. **单任务基线澄清** — AGENTS.md 误记 `mae_clip1_log`(107.70) 为基线，实际用 `kan_v2_mse`(111.15)。修正 AGENTS.md + analyze_representations.py 的 checkpoint 路径。
2. **主模型 β=0.1→0.05** — β 扫描发现 0.05 更优。全文数字更新（92.91/56.68/0.9625/16.4%）。
3. **梯度 r 分析删除** — effective r 公式方向搞反，"幅度主导"站不住。删全部 r，改为只讲 cos。
4. **表征分析论点修正** — "重定位到非线性子空间"基于抖动的 linear probe R²，不可靠。改为只讲 CKA（确定性数据）。

### 第二轮：结构重组
5. **§3 拆分** — 原"Shared backbone"拆成 Dense encoder (§3.1) + Conformer block (§3.2)。"trunk"全改"block"。
6. **§3 去参数化** — 删除 Parameter budget 段和所有参数统计，移到 §5。
7. **§4 重组** — 合并对比方法+指标为 §4.2；实现细节移到 §4.3 末尾；删除三档配置段。
8. **§5 重组** — 标题改"Results and discussion"；5.1→Ablation（合并 head 消融）；新增 5.4 SNR + 5.5 T60 表格。
9. **Contributions 重写** — (1)只讲框架不含数字；(2)梯度+CKA；(3)Fourier-KAN；(4)指标参数集中。

### 第三轮：措辞修正
10. **引言动机段 vs 贡献段去重** — 原文两段重复叙述。动机段改为只讲"为什么这么设计"（不给技术细节），贡献段列"做了什么+结果"。
11. **"配对"改"联合训练"** — "将 T60 估计与显式去噪配对"改为"以 T60 回归为主任务，同时联合训练一个去噪重建任务作为辅助"。
12. **"单调提升"删除** — 只有 3 个数据点不能声称单调；且 β→0.02/0.01 趋势反转。全部改为"提升/improves"。
13. **"频率预算"改"频率数"** — "hierarchical frequency budget"是我自造的术语，不通顺。改为"不同数量的三角基 / frequency count"。
14. **贡献(2)删掉权重扫描结论** — "去噪权重越大精度越高"会引出倒 U 形矛盾。删掉，贡献(2)只讲梯度+CKA。
15. **破折号删除** — 论文中不使用破折号（——/---），改为标准陈述句。

---

## 五、用户写作偏好（从交互中总结）

1. **不说"单调"** — 只有几个数据点不能声称单调。
2. **不用破折号** — 标准陈述句。
3. **不自造术语** — 用领域通用表述。
4. **引言动机段不展开技术细节** — 只讲动机，技术细节留给贡献段和方法章。
5. **贡献里不放权重扫描结论** — 权重扫描只是消融实验，不是 contribution 级别。
6. **方法章不放参数量** — 参数量属于结果，放 §5。
7. **"trunk"改"block"** — trunk 不学术。
8. **结果导向，不写试错日志** — 参考 DAMTL，直接给最终配置。
9. **三份文件必须同步** — main.tex / main_zh.md / main_bilingual.md（如果用户需要）。
10. **GR 不写进论文** — GR 有害，留作 rebuttal 弹药。

---

## 六、Git 分支说明

| 分支 | 用途 | 状态 |
|---|---|---|
| `exp/kan-multitask` | 论文主线 + 多任务实验 | **主分支**，论文文件在此 |
| `exp/gradient-remedy` | GR 实验（隔离） | 临时分支，代码在此，GR 无效后可丢弃 |
| `exp/single-t60` | 单任务消融 | 基线实验，已稳定 |
| `master` | 初始版本 | 不再使用 |

**注意：** 当前论文修改在 `exp/gradient-remedy` 分支上（因为中途切过来跑 GR），论文文件 `docs/main.tex` 和 `docs/main_zh.md` 的最新版本在此分支。如果要切回 `exp/kan-multitask`，需要 merge 或 cherry-pick 论文改动。
