# 条件 GAN 真实 RIR 训练框架

这是一个独立的真实 RIR 条件生成框架，当前不依赖具体 RIR 数据即可完成代码检查。数据就绪后，准备一个 CSV manifest 即可训练。

## 目录

```text
rir_cgan_training/
├── README.md
├── config.json
├── dataset.py          # RIR 读取、统一长度和 T60 条件
├── models.py           # 条件 1D Generator / Critic
├── rir_metrics.py      # Schroeder T60 与候选样本筛选
├── train.py            # WGAN-GP 训练
└── generate.py         # 按目标 T60 生成并筛选 RIR
```

## 数据接口

准备 `manifest.csv`：

```csv
path,t60,room_id,split
/data/rir/a.wav,0.42,room_01,train
/data/rir/b.wav,0.87,room_02,val
```

当前框架约定：

- 单通道 WAV；
- 统一采样率由配置指定；
- `t60` 为秒；
- 训练/验证建议按房间划分，而不是随机按 RIR 划分；
- 过长 RIR 尾部裁剪，过短 RIR 尾部补零。

## 训练

```bash
python train.py \
  --manifest /path/to/manifest.csv \
  --config config.json \
  --out_dir runs/exp001
```

没有数据时可先运行：

```bash
python train.py --help
python generate.py --help
```

## 生成与筛选

```bash
python generate.py \
  --checkpoint runs/exp001/checkpoints/latest.pt \
  --config config.json \
  --out_dir generated/exp001 \
  --t60 0.4 0.8 1.2 \
  --num_per_t60 100
```

生成阶段会记录目标 T60、重新估计的 T60、误差和是否通过筛选。当前筛选器是保守的第一版，真实数据到位后需要用验证集校准阈值。

## 当前明确的后续工作

1. 用真实 RIR 确认采样率、有效长度和标签格式；
2. 校准 `config.json` 的 `rir_length`、T60 范围和筛选阈值；
3. 先跑小规模训练，检查生成 RIR 的 T60 可控性和模式崩溃；
4. 通过验证后，再接入语音卷积和现有多任务数据生成流程。

