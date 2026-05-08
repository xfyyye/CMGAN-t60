# CMGAN-t60 Git 分支与 8 卡实验运行整理方案

## 目标

当前项目 `CMGAN-t60` 后续会同时维护多个代码版本，并在同一台 8 GPU 机器上并行跑实验。

实验资源设定：

```text
一共 8 张 GPU
每个实验任务占用 2 张 GPU
同一时间最多并行 4 个实验任务
任务使用 nohup 后台管理
所有实验结果统一保存到 runs/ 文件夹
每个实验目录必须能看出使用了哪个 config
```

核心原则：

```text
大代码结构差异用 Git branch 管理
小实验变量用 config 管理
多分支同机并行用 git worktree 管理
每个实验用 2 张 GPU，通过 CUDA_VISIBLE_DEVICES + torchrun 启动
每次实验必须保存 config、git commit、日志、模型输出到 runs/
```

---

## 一、Git 分支设计

当前仓库主分支是：

```text
master
```

不是 `main`。

建议分支结构如下：

```text
master
    当前稳定多任务版本，作为原始 baseline

exp/single-t60
    单任务 T60 估计版本
    用于把原来的多任务模型改造成只预测 T60 的模型

exp/single-t60-kan
    单任务 T60 + KAN head 版本
    在 exp/single-t60 的基础上，把原 MLP regression head 替换成 KAN / Fourier KAN head
```

不建议为每个 loss 单独开 Git 分支，例如不建议这样：

```text
exp/kan-mse
exp/kan-mae
exp/kan-huber
```

MSE、MAE、Huber、不同学习率、不同 batch size、不同 KAN 参数都应该优先放进 config 文件。

---

## 二、Git 操作与含义

### 1. 查看当前状态

```bash
git status
```

作用：查看当前在哪个分支、有没有未提交的修改、新增文件或删除文件。

### 2. 给当前多任务版本打 tag

```bash
git tag v0.1-multitask-baseline
```

作用：给当前 `master` 代码版本打一个固定快照标签，表示这是原始多任务 baseline。

### 3. 推送 master

```bash
git push origin master
```

作用：把本地 `master` 分支推送到 GitHub 远程仓库。

### 4. 推送 tag

```bash
git push origin v0.1-multitask-baseline
```

作用：把本地 tag 上传到 GitHub，方便之后任何机器都能 checkout 到这个固定版本。

### 5. 创建单任务分支

```bash
git checkout -b exp/single-t60
```

作用：从当前 `master` 创建一个新分支 `exp/single-t60`，并立即切换到这个分支。

这个分支用于完成：

```text
多任务输出头 -> 单任务 T60 输出头
多任务 loss -> T60 单任务 loss
多任务 metric -> T60 metric
数据读取和 label 对齐到 T60 估计任务
```

### 6. 推送单任务分支

```bash
git push -u origin exp/single-t60
```

作用：把本地 `exp/single-t60` 分支推送到 GitHub，并建立本地分支和远程分支的追踪关系。

之后在这个分支上可以直接使用：

```bash
git push
git pull
```

### 7. 提交单任务代码改动

```bash
git add .
```

作用：把当前修改加入 Git 暂存区。

```bash
git commit -m "Convert model to single-task T60 estimation"
```

作用：提交一次 Git 版本，说明这次改动是把模型改造成单任务 T60 估计。

```bash
git push
```

作用：把当前分支的新提交推送到 GitHub。

### 8. 从单任务分支创建 KAN 分支

确保当前在 `exp/single-t60`：

```bash
git checkout exp/single-t60
```

作用：切换到单任务 T60 分支。

然后创建 KAN 分支：

```bash
git checkout -b exp/single-t60-kan
```

作用：从单任务 T60 分支创建 `exp/single-t60-kan` 分支。

这个分支用于完成：

```text
MLP regression head -> KAN / Fourier KAN regression head
保留单任务 T60 框架
支持通过 config 切换 MLP / KAN / Fourier KAN head
```

推送：

```bash
git push -u origin exp/single-t60-kan
```

作用：把 KAN 分支推送到 GitHub。

---

## 三、推荐代码目录整理

建议整理成下面结构：

```text
CMGAN-t60/
├── configs/
│   └── t60_single/
│       ├── mlp_mse.yaml
│       ├── mlp_mae.yaml
│       ├── mlp_huber.yaml
│       ├── kan_mse.yaml
│       ├── kan_mae.yaml
│       ├── kan_huber.yaml
│       ├── fourier_kan_mse.yaml
│       └── fourier_kan_huber.yaml
│
├── models/
│   ├── cmgan.py
│   ├── heads/
│   │   ├── mlp_t60_head.py
│   │   ├── kan_t60_head.py
│   │   └── fourier_kan_t60_head.py
│
├── losses/
│   └── t60_loss.py
│
├── scripts/
│   ├── run_nohup_2gpu.sh
│   ├── run_4tasks_8gpu.sh
│   └── kill_run.sh
│
├── runs/
│   └── automatically_generated_experiment_dirs/
│
├── train.py
├── evaluate.py
├── README.md
└── .gitignore
```

---

## 四、config 设计要求

每个 config 需要能完整描述一次实验。

例如：

```yaml
experiment:
  name: fourier_kan_huber
  task: single_t60
  output_root: runs

model:
  backbone: cmgan
  head_type: fourier_kan
  t60_output_dim: 1

kan:
  num_frequencies: 16
  hidden_dim: 256
  dropout: 0.1

loss:
  name: huber
  delta: 1.0

train:
  epochs: 100
  batch_size: 32
  lr: 0.0001
  seed: 42
  num_workers: 8

distributed:
  enabled: true
  backend: nccl
```

要求：

```text
experiment.name 必须存在
experiment.output_root 默认是 runs
loss.name 支持 mse / mae / huber
model.head_type 支持 mlp / kan / fourier_kan
训练启动后必须把当前 config 复制到对应 runs 子目录中
```

---

## 五、runs 实验保存规范

所有实验必须保存到：

```text
runs/
```

每个实验单独一个目录，目录名需要包含：

```text
时间戳
实验名
config 名
```

推荐格式：

```text
runs/20260430_153012_fourier_kan_huber__cfg_fourier_kan_huber/
```

每个实验目录内部建议包含：

```text
runs/20260430_153012_fourier_kan_huber__cfg_fourier_kan_huber/
├── config.yaml
├── git_info.txt
├── nohup.log
├── train.log
├── metrics.json
├── checkpoints/
│   ├── latest.pt
│   └── best.pt
└── tensorboard/
```

其中：

- `config.yaml`：保存本次实验实际使用的 config 副本。
- `git_info.txt`：保存当前分支名、commit id、是否有未提交修改。
- `nohup.log`：保存 nohup 启动后的标准输出和错误输出。
- `train.log`：保存训练代码内部记录的训练日志。
- `metrics.json`：保存最终指标，例如 val MAE、val RMSE、test MAE、test RMSE。
- `checkpoints/`：保存模型权重。

`git_info.txt` 内容示例：

```text
branch: exp/single-t60-kan
commit: a3f91c8d2e4bxxxx
config: configs/t60_single/fourier_kan_huber.yaml
dirty: false
```

---

## 六、.gitignore 要求

`runs/`、模型权重、日志、数据不要提交到 GitHub。

`.gitignore` 建议包含：

```gitignore
runs/
logs/
outputs/
checkpoints/
wandb/
tensorboard/
data/

*.pt
*.pth
*.ckpt
*.npy
*.pkl
*.log
```

但 config 文件必须提交：

```text
configs/**/*.yaml
```

---

## 七、同机多分支并行：git worktree

如果要同时跑 `exp/single-t60` 和 `exp/single-t60-kan`，不要在同一个目录反复 `git checkout`。

推荐使用 `git worktree`。

在原始仓库目录下执行：

```bash
git worktree add ../CMGAN-single-t60 exp/single-t60
```

作用：创建一个新目录 `../CMGAN-single-t60`，这个目录固定对应 `exp/single-t60` 分支。

```bash
git worktree add ../CMGAN-kan exp/single-t60-kan
```

作用：创建一个新目录 `../CMGAN-kan`，这个目录固定对应 `exp/single-t60-kan` 分支。

最终目录结构：

```text
CMGAN-t60/
    原始仓库目录，可保留 master

CMGAN-single-t60/
    单任务 MLP baseline 代码目录，对应 exp/single-t60

CMGAN-kan/
    KAN 代码目录，对应 exp/single-t60-kan
```

这样可以同时在不同目录启动不同分支的实验。

---

## 八、2 GPU nohup 启动脚本设计

需要新增脚本：

```text
scripts/run_nohup_2gpu.sh
```

推荐功能：

```bash
bash scripts/run_nohup_2gpu.sh   configs/t60_single/fourier_kan_huber.yaml   0,1
```

含义：

```text
使用 configs/t60_single/fourier_kan_huber.yaml
占用 GPU 0 和 GPU 1
用 nohup 后台启动
实验输出保存到 runs/
```

脚本建议实现如下：

```bash
#!/usr/bin/env bash
set -e

CONFIG=$1
GPUS=$2

if [ -z "$CONFIG" ] || [ -z "$GPUS" ]; then
  echo "Usage: bash scripts/run_nohup_2gpu.sh <config_path> <gpu_ids>"
  echo "Example: bash scripts/run_nohup_2gpu.sh configs/t60_single/fourier_kan_huber.yaml 0,1"
  exit 1
fi

mkdir -p runs

CONFIG_BASENAME=$(basename "$CONFIG" .yaml)
EXP_NAME=$(python - <<PY
import yaml
with open("$CONFIG", "r") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("experiment", {}).get("name", "$CONFIG_BASENAME"))
PY
)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="runs/${TIMESTAMP}_${EXP_NAME}__cfg_${CONFIG_BASENAME}"

mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/checkpoints"

cp "$CONFIG" "$RUN_DIR/config.yaml"

DIRTY=$(test -n "$(git status --porcelain)" && echo true || echo false)

{
  echo "branch: $(git branch --show-current)"
  echo "commit: $(git rev-parse HEAD)"
  echo "config: $CONFIG"
  echo "gpus: $GPUS"
  echo "dirty: $DIRTY"
} > "$RUN_DIR/git_info.txt"

NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')

CUDA_VISIBLE_DEVICES=$GPUS nohup torchrun   --standalone   --nproc_per_node=$NUM_GPUS   train.py   --config "$CONFIG"   --run_dir "$RUN_DIR"   > "$RUN_DIR/nohup.log" 2>&1 &

echo "Started experiment:"
echo "  config: $CONFIG"
echo "  gpus: $GPUS"
echo "  run_dir: $RUN_DIR"
echo "  pid: $!"
```

注意：这要求 `train.py` 支持：

```bash
--config
--run_dir
```

并且支持 `torchrun` / DDP 多卡训练。

---

## 九、8 GPU 同时启动 4 个实验任务

新增脚本：

```text
scripts/run_4tasks_8gpu.sh
```

每个任务占用 2 张 GPU：

```text
任务 1：GPU 0,1
任务 2：GPU 2,3
任务 3：GPU 4,5
任务 4：GPU 6,7
```

脚本示例：

```bash
#!/usr/bin/env bash
set -e

bash scripts/run_nohup_2gpu.sh configs/t60_single/mlp_mse.yaml 0,1
bash scripts/run_nohup_2gpu.sh configs/t60_single/mlp_huber.yaml 2,3
bash scripts/run_nohup_2gpu.sh configs/t60_single/fourier_kan_mse.yaml 4,5
bash scripts/run_nohup_2gpu.sh configs/t60_single/fourier_kan_huber.yaml 6,7
```

运行：

```bash
bash scripts/run_4tasks_8gpu.sh
```

---

## 十、train.py 需要支持的参数

coding agent 需要检查并补充 `train.py`：

```bash
python train.py --config xxx.yaml --run_dir runs/xxx
```

必须支持：

```text
--config
    指定实验配置文件路径

--run_dir
    指定本次实验输出目录
```

训练代码内部需要使用 `run_dir` 保存：

```text
train.log
metrics.json
checkpoints/latest.pt
checkpoints/best.pt
tensorboard/
```

不要在代码中写死输出目录。

错误示例：

```python
save_dir = "checkpoints/"
```

推荐：

```python
save_dir = os.path.join(args.run_dir, "checkpoints")
```

---

## 十一、DDP / torchrun 要求

因为每个任务使用 2 张 GPU，推荐使用：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py ...
```

注意：

```text
CUDA_VISIBLE_DEVICES=0,1
```

表示当前任务只能看到物理 GPU 0 和 1。

在程序内部，这两张卡会变成：

```text
cuda:0
cuda:1
```

不是原来的物理编号。

`train.py` 中需要正确读取：

```text
LOCAL_RANK
RANK
WORLD_SIZE
```

基本逻辑：

```python
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
```

并用 DDP 包装模型：

```python
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank
)
```

DataLoader 需要使用：

```python
DistributedSampler
```

保证两张卡不会重复读取完全相同的数据。

---

## 十二、实验运行示例

### 1. 跑单个 2 GPU 实验

```bash
bash scripts/run_nohup_2gpu.sh configs/t60_single/fourier_kan_huber.yaml 0,1
```

作用：

```text
用 GPU 0,1 跑 fourier_kan_huber 实验
后台运行
输出保存到 runs/时间戳_fourier_kan_huber__cfg_fourier_kan_huber/
```

### 2. 同时跑 4 个实验

```bash
bash scripts/run_4tasks_8gpu.sh
```

作用：

```text
GPU 0,1 跑实验 1
GPU 2,3 跑实验 2
GPU 4,5 跑实验 3
GPU 6,7 跑实验 4
```

### 3. 查看某个实验日志

```bash
tail -f runs/20260430_153012_fourier_kan_huber__cfg_fourier_kan_huber/nohup.log
```

作用：实时查看该实验的 nohup 日志。

### 4. 查看 GPU 使用情况

```bash
nvidia-smi
```

作用：查看当前每张 GPU 的显存占用和运行进程。

---

## 十三、分支与实验对应建议

### MLP baseline 实验

代码分支：

```text
exp/single-t60
```

配置：

```text
configs/t60_single/mlp_mse.yaml
configs/t60_single/mlp_mae.yaml
configs/t60_single/mlp_huber.yaml
```

运行目录：

```text
CMGAN-single-t60/
```

### KAN 实验

代码分支：

```text
exp/single-t60-kan
```

配置：

```text
configs/t60_single/kan_mse.yaml
configs/t60_single/kan_mae.yaml
configs/t60_single/kan_huber.yaml
configs/t60_single/fourier_kan_mse.yaml
configs/t60_single/fourier_kan_huber.yaml
```

运行目录：

```text
CMGAN-kan/
```

---

## 十四、推荐的一次 8 卡实验安排

```text
GPU 0,1:
    branch: exp/single-t60
    config: configs/t60_single/mlp_mse.yaml

GPU 2,3:
    branch: exp/single-t60
    config: configs/t60_single/mlp_huber.yaml

GPU 4,5:
    branch: exp/single-t60-kan
    config: configs/t60_single/fourier_kan_mse.yaml

GPU 6,7:
    branch: exp/single-t60-kan
    config: configs/t60_single/fourier_kan_huber.yaml
```

如果跨分支同时跑，建议分别在两个 worktree 目录里执行：

```bash
cd ../CMGAN-single-t60
bash scripts/run_nohup_2gpu.sh configs/t60_single/mlp_mse.yaml 0,1
bash scripts/run_nohup_2gpu.sh configs/t60_single/mlp_huber.yaml 2,3
```

```bash
cd ../CMGAN-kan
bash scripts/run_nohup_2gpu.sh configs/t60_single/fourier_kan_mse.yaml 4,5
bash scripts/run_nohup_2gpu.sh configs/t60_single/fourier_kan_huber.yaml 6,7
```

---

## 十五、coding agent 需要完成的任务清单

### Git / 分支相关

```text
1. 保留 master 作为稳定多任务 baseline
2. 创建 exp/single-t60 分支
3. 创建 exp/single-t60-kan 分支
4. 不要为每个 loss 新开分支
5. 确保 configs/ 被 Git 管理
6. 确保 runs/、checkpoints/、日志、权重不被 Git 管理
```

### config 相关

```text
1. 增加 configs/t60_single/ 目录
2. 准备 mlp_mse.yaml
3. 准备 mlp_mae.yaml
4. 准备 mlp_huber.yaml
5. 准备 kan_mse.yaml
6. 准备 kan_mae.yaml
7. 准备 kan_huber.yaml
8. 准备 fourier_kan_mse.yaml
9. 准备 fourier_kan_huber.yaml
10. 每个 config 必须包含 experiment.name
11. 每个 config 必须包含 loss.name
12. 每个 config 必须包含 model.head_type
```

### 训练入口相关

```text
1. train.py 支持 --config
2. train.py 支持 --run_dir
3. 所有输出路径基于 run_dir
4. 每次训练保存 metrics.json
5. 每次训练保存 best.pt 和 latest.pt
6. 支持 torchrun / DDP 双卡训练
7. DDP 下使用 DistributedSampler
8. 只在 rank 0 保存日志、config、checkpoint、metrics
```

### 运行脚本相关

```text
1. 新增 scripts/run_nohup_2gpu.sh
2. 新增 scripts/run_4tasks_8gpu.sh
3. run_nohup_2gpu.sh 自动创建 runs 子目录
4. run_nohup_2gpu.sh 自动复制 config 到 run_dir
5. run_nohup_2gpu.sh 自动记录 git branch 和 commit
6. run_nohup_2gpu.sh 自动把 nohup 输出保存到 run_dir/nohup.log
7. run_4tasks_8gpu.sh 按 0,1 / 2,3 / 4,5 / 6,7 分配 GPU
```

---

## 十六、最终执行原则

最终项目应该满足：

```text
1. 代码版本清楚：
   master 保存多任务 baseline
   exp/single-t60 保存单任务 T60
   exp/single-t60-kan 保存 KAN 版本

2. 实验变量清楚：
   MSE / MAE / Huber / KAN 参数都由 config 控制

3. 运行方式清楚：
   每个 nohup 任务占 2 张 GPU
   8 张 GPU 同时跑 4 个实验

4. 结果保存清楚：
   所有实验都在 runs/
   每个 run 目录名包含实验名和 config 名
   每个 run 内部保存 config.yaml、git_info.txt、nohup.log、metrics.json、checkpoints/

5. 可复现：
   每个实验都能通过 git commit + config.yaml 还原
```
