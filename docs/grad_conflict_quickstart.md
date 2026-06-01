# 多任务梯度冲突实验 · 新手执行指南

把 master 派生的 **MLP-head 多任务**分支（`exp/grad-angle-probe`）和
single-t60-kan 派生的 **KAN-head 多任务**分支（`exp/kan-multitask`）一起跑起来，
观察去噪 / T60 两条任务的梯度夹角与量级失衡，再读懂结果。

前提：已经有一套能成功跑 single-t60-kan 的环境（conda env: `demucs_xxn`，
torch + torchaudio + swanlab + PyYAML 都齐全），且数据集
`T60_Dataset_v7_4s` 在本机可访问。

---

## 1. 准备：拉两份独立工作目录

两个分支彼此独立，**最简单**的做法就是各 clone 一份到不同目录：

```bash
mkdir -p ~/multitask-experiments && cd ~/multitask-experiments

# MLP-head 多任务 (master 派生)
git clone -b exp/grad-angle-probe https://github.com/xfyyye/CMGAN-t60.git mlp-multitask

# KAN-head 多任务 (single-t60-kan 派生)
git clone -b exp/kan-multitask    https://github.com/xfyyye/CMGAN-t60.git kan-multitask

ls
# mlp-multitask/   kan-multitask/
```

> 如果你想省磁盘，也可以在已有的 single-kan-t60 仓库里用
> `git worktree add ../mlp-multitask origin/exp/grad-angle-probe` 这种方式，
> 但 worktree 对新手心智负担更高，**推荐直接 clone 两份**。

确认两边都拉到了：

```bash
cd ~/multitask-experiments/mlp-multitask && git log --oneline -1
# 应该看到: 3f2170d 探针指标同时打印到 stdout

cd ~/multitask-experiments/kan-multitask && git log --oneline -1
# 应该看到: d07db3f 探针指标同时打印到 stdout
```

## 2. 数据集路径

两个分支的取数方式不一样：

| 分支 | 默认路径配置位置 | 怎么覆盖 |
|---|---|---|
| `mlp-multitask` | shell 变量 `T60_DATASET_ROOT`（在 `train_grad_probe.sh` 里改，或运行时传 `-d`） | `-d /path/to/T60_Dataset_v7_4s` |
| `kan-multitask` | 4 个 yaml 里 `data.dataset_root` | `-d /path/...` 或直接编辑 yaml |

如果数据集放在 `/data/T60_Dataset_v7_4s`：

```bash
# MLP 那侧 (编辑一次，4 个 wrapper 都生效)
sed -i 's|^T60_DATASET_ROOT.*|T60_DATASET_ROOT="${T60_DATASET_ROOT:-/data/T60_Dataset_v7_4s}"|' \
    ~/multitask-experiments/mlp-multitask/train_grad_probe.sh

# KAN 那侧 (4 个 yaml 都改)
sed -i 's|^  dataset_root:.*|  dataset_root: /data/T60_Dataset_v7_4s|' \
    ~/multitask-experiments/kan-multitask/configs/t60_multitask/*.yaml
```

不想改文件、想运行时临时传也可以，每个脚本都接受 `-d`。

## 3. 30 秒 sanity check（强烈建议）

正式 nohup 之前，**先确认探针真的会打印** —— 跑 1 分钟然后 Ctrl+C：

```bash
cd ~/multitask-experiments/kan-multitask
conda activate demucs_xxn
CUDA_VISIBLE_DEVICES=0,1 python train_multitask.py \
    --config configs/t60_multitask/kan_mse_w_eq.yaml --train_only \
    2>&1 | head -80
# 按 Ctrl+C 中断
```

应该在前 1 分钟里看到至少 1 行类似的探针输出：

```
[probe step=0] encoder:cos=+0.038,r=0.82 tscb_1:cos=-0.012,r=0.91 tscb_2:cos=-0.105,r=1.13
```

看到了就 OK，没看到说明 swanlab 之外还有别的问题，先排掉再 nohup。

## 4. nohup 启动

约定：8 块 GPU 的机器，每个实验吃 2 卡。一次跑 4 个实验（一个分支的 4 个权重）。
两个分支错开跑，第一波结束再跑第二波。

### 第一波：KAN 多任务 4 个权重

```bash
cd ~/multitask-experiments/kan-multitask

nohup ./scripts/multitask_kan_w_eq.sh       -g 0,1 > nohup_kan_w_eq.log       2>&1 &
nohup ./scripts/multitask_kan_w_t60.sh      -g 2,3 > nohup_kan_w_t60.log      2>&1 &
nohup ./scripts/multitask_kan_w_denoise.sh  -g 4,5 > nohup_kan_w_denoise.log  2>&1 &
nohup ./scripts/multitask_kan_w_balanced.sh -g 6,7 > nohup_kan_w_balanced.log 2>&1 &

jobs -l   # 应看到 4 个进程
nvidia-smi  # 确认 GPU 0-7 都被吃满
```

### 第二波：MLP 多任务 4 个权重（等第一波 4 个都结束）

```bash
cd ~/multitask-experiments/mlp-multitask

nohup ./scripts/grad_probe_w_eq.sh       -g 0,1 > nohup_mlp_w_eq.log       2>&1 &
nohup ./scripts/grad_probe_w_t60.sh      -g 2,3 > nohup_mlp_w_t60.log      2>&1 &
nohup ./scripts/grad_probe_w_denoise.sh  -g 4,5 > nohup_mlp_w_denoise.log  2>&1 &
nohup ./scripts/grad_probe_w_balanced.sh -g 6,7 > nohup_mlp_w_balanced.log 2>&1 &
```

> 单波 4 个并行运行时间预估：T60_Dataset_v7 训练集 40k 样本，
> batch=32 + 2GPU，单 epoch 约 8-12 分钟；100 epoch 早停一般 30-50 epoch
> 收敛 → 单实验 4-8 小时，4 个并行 ≈ 4-8 小时一波。

## 5. 训练过程中怎么观察梯度冲突

有两条路：**stdout 快照** 和 **SwanLab 时间序列**。

### 5.1 stdout（适合实时瞄一眼）

探针每 20 个 train step 触发一次（`probe.log_every`，可在 yaml 改），
每次往 nohup 日志里塞一行：

```
[probe step=320] encoder:cos=+0.124,r=0.45 tscb_1:cos=-0.031,r=0.51 tscb_2:cos=-0.183,r=0.62
```

字段含义：

| 字段 | 含义 |
|---|---|
| `cos` | 该 group 上 denoise 梯度和 t60 梯度的余弦相似度 |
| `r` | `‖g_denoise‖ / ‖g_t60‖`，量级失衡度 |
| group | `encoder` / `tscb_1` / `tscb_2`（KAN 默认 2 层；MLP 默认 4 层时还有 tscb_3/4）|

只看探针行：

```bash
tail -f nohup_kan_w_eq.log | grep '\[probe'
```

汇总最近 100 条做粗略平均（看趋势）：

```bash
grep '\[probe' nohup_kan_w_eq.log | tail -100 | \
    awk -F'cos=|,r=' '{print $2, $4, $6, $8}'
```

### 5.2 SwanLab（适合做对比和论文图）

4 个 KAN 实验都在项目 `T60_KAN_MultiTask`，4 个 MLP 实验都在项目
`T60_GradAngleProbe`。打开 SwanLab → 选项目 → 看以下面板：

- **冲突方向**：`grad/encoder/cos`, `grad/tscb_1/cos`, `grad/tscb_2/cos`
- **量级失衡**：`grad/*/ratio_d_over_t`
- **原始量级**：`grad/*/norm_denoise`, `grad/*/norm_t60`

同项目内 4 条曲线自动叠加，可直接对比 4 种 α/β 的差异。
跨项目（MLP vs KAN）需要分别截图或导出 CSV 自己画。

横轴是 `global_step`，不是 epoch。4 条曲线总步数差不多。

## 6. 可能出现的现象 · 理论解释 · 怎么应对

下面把常见的几种 pattern 列出来，每种给一句话理论背景和一个可执行的下一步。

### 6.1 cos 一直 ≈ 0（|cos| < 0.05）

**现象**：两个任务的梯度几乎正交，互不影响。

**理论**：在 shared encoder 上，任务梯度方向高维近乎随机 → 余弦接近 0 是
"任务无关"。这相当于**意外参数共享**：两条任务各自学各自的，只是恰好住在
同一组参数里。

**判定**：如果 T60 单任务的精度 ≈ 多任务 T60 精度，确认无传递。

**应对**：多任务在这里**没增益也没害**；要么放弃辅助任务（省算力），要么换一个跟 T60 更相关的辅助任务（比如直接预测 DRR / C50）。

---

### 6.2 cos 持续为正（≥ 0.2）

**现象**：去噪和 T60 梯度方向一致，互相帮助。

**理论**：**正向迁移**（positive transfer）。共享 backbone 学到的特征对两条
任务都有用。这是多任务学习"理想中"的场景。
参考：Crawshaw 2020 的多任务学习综述。

**判定**：多任务 T60 精度应该明显优于 single-task T60。

**应对**：保留多任务设置；可以加大 batch / 加深 backbone 巩固特征。

---

### 6.3 cos 持续为负（≤ -0.1）

**现象**：两任务在抢方向，一方的更新会部分抵消另一方。

**理论**：**梯度冲突**。这是 PCGrad (Yu et al. NeurIPS 2020)、CAGrad、
MGDA-UB 等论文的核心动机 —— 把冲突分量投影掉再合成更新。

**判定**：T60 精度可能不升反降（与 single-task 比）。

**应对**：
1. 尝试 PCGrad：把 t60 梯度在 denoise 梯度上的负投影分量减掉
2. 或降低权重冲突更严重那个任务的权重（看 `r` 决定降哪个）
3. 或在 backbone 顶端**任务特定分支**化（最后 1 层 TSCB 拆成 2 个 head）

---

### 6.4 cos 先负后正（或先正后负）

**现象**：训练前期 cos < 0，后期慢慢爬到 > 0；或者反过来。

**理论**：**表征对齐过程**。模型早期还没学到通用特征，任务梯度自然冲突；
随着 backbone 学到"既能去噪又能估 T60"的中性表征，方向逐渐对齐。反过来
的情况通常是模型走进了某个对一方任务过拟合的 basin。

**判定**：通常出现在 lr warmup 阶段，正常现象。

**应对**：不必干预；只在长期保持负值时才需要 PCGrad。

---

### 6.5 ratio_d_over_t >> 1（如 > 10）

**现象**：去噪梯度量级压倒性地大于 T60 梯度。

**理论**：**量级失衡**。即便 cos > 0，优化器实际更新方向也基本就是
denoise 的方向，T60 只贡献微弱的扰动。这是 GradNorm (Chen et al. ICML 2018)
的motivation：自适应调整任务权重让各任务梯度量级一致。

**判定**：看 T60 val loss 下降速度是否远慢于 single-task T60。

**应对**：
1. 增大 `beta`（yaml 里 `loss.beta: 1.0 → 5.0`）
2. 或用 GradNorm 自动调整
3. 或在 yaml 里降 `w_mag` / `w_ri` 让 denoise 自身小一点

---

### 6.6 ratio_d_over_t << 1（如 < 0.1）

**现象**：T60 梯度反过来压倒 denoise。

**理论**：通常意味着 T60 任务（标量回归）的梯度天生比谱图回归小，或者
denoise loss 已经收敛到平台期。

**应对**：如果 T60 是主任务，这本来就是想要的；如果是为了诊断转移效应，
适当降 `beta` 或升 `alpha` 让两者量级相当。

---

### 6.7 encoder cos > tscb_2 cos（深层冲突更严重）

**现象**：浅层 cos ≈ +0.2，深层 cos ≈ -0.1。

**理论**：**hard parameter sharing 的经典模式** —— 低层特征（频谱基函数）
对两任务都通用，但高层语义（去噪 vs RT60 估计）开始分化，硬共享反而拖后腿。

**应对**：把模型改成 **shared encoder + task-specific TSCB tails** —— 即
encoder + TSCB_1 共享，TSCB_2 拆成两个独立模块。

---

### 6.8 KAN 分支 vs MLP 分支的差异

可能看到的几种情况：

- **backbone cos 几乎相同**：head 选择对 backbone 上的任务关系无影响 →
  head 只决定 T60 任务自身的拟合能力，不影响多任务动力学。这是大部分情况。
- **KAN 的 backbone cos 更正**：KAN head 在 head 内部"吸收"了更多 T60 特异性，
  反过来让流回 backbone 的梯度更"通用"。如果观察到这个，是 KAN 在多任务场景
  下的一个意外优点。
- **KAN 的 cos 更负**：相反情况，KAN head 把太多任务特异性塞回 backbone。

不论哪种，最终 T60 精度才是 ground truth；cos 只是过程指标。

---

### 6.9 4 个权重组合的 cos 几乎重合

**现象**：`w_eq` (1/1)、`w_t60` (0.1/1)、`w_denoise` (1/0.1)、`w_balanced` (0.5/0.5)
的 cos 曲线挤在一起。

**理论**：cos 数学上是**尺度无关**的（α 缩放 g_denoise 不改变方向），所以
理论上 cos 应该几乎相同 —— 如果真的重合，说明你的实现没 bug。

**反例**：如果 cos 在 4 种权重下差异很大，说明权重通过改变优化轨迹间接改变了
模型走到的 basin，从而改变了任务关系 —— 这本身是个有趣发现。

**对应的 ratio_d_over_t**：会跟随 α/β 等比缩放 —— `r(w_denoise) ≈ 100 × r(w_t60)`
是预期；如果不是，说明 loss 里有非线性项干扰。

## 7. 训练完成后

每个实验会在 `runs/<experiment_name>/` 留下：

```
best_model.pth          # 早停保存的最优 ckpt（按 t60_loss 选）
training_history.json   # 每 epoch 的 train/val loss
```

跑 4 个 test split 拿 T60 精度数字：

```bash
cd ~/multitask-experiments/kan-multitask
for w in eq t60 denoise balanced; do
    python test_multitask.py \
        --config configs/t60_multitask/kan_mse_w_${w}.yaml \
        --model_path runs/kan_multitask_w_${w}/best_model.pth \
        --save_dir runs/kan_multitask_w_${w}/test
done
```

MLP 分支用 `test.py`（不是 test_multitask.py），具体看那边 README。

最后做一张对比表：单任务 T60 vs 4 种多任务（MLP）vs 4 种多任务（KAN），
配合 cos / ratio 的 swanlab 截图，就是一份完整的梯度冲突实验报告。

## 8. 常见坑

| 症状 | 原因 | 修法 |
|---|---|---|
| `ModuleNotFoundError: torch` | 没 activate conda env | `conda activate demucs_xxn` |
| `数据集 split 目录不存在` | 路径没改对 | 编辑 yaml 或加 `-d` 参数 |
| `多任务训练需要纯混响 target，但缺失` | 数据集少了 `*_denoised.wav` | 确认是 v7_4s 完整版 |
| SwanLab 初始化失败 | 网络问题或 token 没配 | 不影响训练，stdout 还有探针行 |
| 进度条把探针行刷掉 | 终端宽度不够 | 用 `tail -f xx.log \| grep probe` 过滤 |
| 4 个并行 GPU 利用率不均 | DataParallel 默认分卡 | 正常，DP 主卡负载略高 |
| ratio_d_over_t 数值爆炸 (>1e6) | AMP 下梯度被 GradScaler 缩放 | 探针内部已除掉 scale，理论上不会，看到了请保留日志反馈 |

## 9. 结论怎么写

最有用的三组对比：

1. **单任务 T60 (single-t60-kan 分支) vs 多任务 T60 (这次的两个分支)**
   —— 多任务到底有没有帮 T60
2. **MLP head 多任务 vs KAN head 多任务**
   —— head 选择在多任务场景下的表现差异
3. **4 种 α/β 内的对比**
   —— 看权重对 T60 精度和梯度行为的影响曲线

3 个对比加上 cos / ratio 的曲线，就能讲清楚 "多任务对 T60 有没有帮助 + 哪种
head 更适配 + 权重该怎么选"。
