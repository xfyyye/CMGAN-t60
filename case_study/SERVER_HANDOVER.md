# Case Study 服务器交接（2026-10-21）

## 当前状态与阅读顺序

本次任务为三个已选 AIR 实测 RIR 的独立 Case Study。脚本已准备，仅完成静态检查，**尚未合成、加载 checkpoint、推理或验证运行环境**。完整协议和四组权重映射见 [README.md](README.md)、[config.json](config.json)。不能把“代码已准备”写成“实验已完成”。

依次读仓库 `docs/HANDOVER.md`、`AGENTS.md` 顶部当前约定，再读本文件和 [SERVER_PROMPT.md](SERVER_PROMPT.md)。本文件补充 10 月 21 日后续已确认的 Case Study 协议；历史 HANDOVER 中“先恢复 Test 2/4 房间映射”已不是本任务前置条件。用户已决定另建这三个场景，不要求恢复旧测试集映射。未重新读取飞书，仅使用交接快照和用户后续决定，不推测新要求。

主线为 `exp/kan-multitask`。远端提交 `39b7499` 已独立迁入 `rir_cgan_training/`；那是尚未完成的 RIR 增广方向，**不属于这次 Case Study，不启动它**。不整分支合并历史 `exp/gradient-remedy`，不恢复表征分析、梯度探针或相关机制论断。本机稿件和图表还有独立的未提交更新，本次只推送 Case Study 目录，服务器不能假定论文已同步到本机最新状态；本任务不改三份稿件及原训练代码。

## 工作目录和数据摆放

`case_study/` 是代码、配置及独立输出所在的子项目；**命令从整个仓库根目录执行**。评估依赖根目录 `dataset.py`、`utils.py`、`models/` 等，不能只复制 Case Study 文件夹到服务器。

```text
/服务器项目路径/CMGAN-t60/          # 终端、Codex 的工作目录
  case_study/
    config.json                  # 版本化协议，保留本机原料路径供追溯
    config.server.local.json     # 服务器配置副本，Git 忽略
    outputs/<独立版本>/          # 生成音频和预测，Git 忽略
  configs/                       # 四组冻结模型配置
  runs/                          # 四组指定 best_model.pth

/服务器数据路径/rawdataset/        # 仓库外；本机对应 /Users/xiexinni/dataset/rawdataset
  Dry/TIMIT_90_test_dry/
    test8s_001.wav ... test8s_090.wav
  Noise/noise92_10_test_noise_unseen/
    white.wav
    babble.wav
  RIR/OpenSLR28_307_test_realrir/
    T60_0.454_000.wav
    T60_0.773.wav
    T60_1.138.wav
    t60_calculation_results.csv
```

以上路径中的“服务器项目路径／服务器数据路径”是占位，必须使用服务器真实位置。原料不随 Git 推送；服务器可复用现存一致原料，也可由用户另行传输这份最小集合：90 个干声、2 个完整噪声、3 个双通道 RIR 和 1 个完整 CSV，共 96 个文件。压缩包 `rirs_noises.zip` 本身不够，它不包含这里的干声、历史重命名映射及全部所需噪声原料。

保持原始层级、文件名和内容。三个 RIR 保留双通道，合成脚本内取通道索引 0；不要提前转单声道。干声必须是原 8 s／16 kHz／单通道／128000 点 WAV，不能用现有 4 s 混合测试音频替代。两份噪声必须完整传输，长度会影响随机截取位置。标签 CSV 必须保留完整 325 条数据记录和原顺序（含表头共 326 物理行）；不能只摘三个案例，因为脚本检查物理行 78、109、123。

[ASSET_INDEX.json](ASSET_INDEX.json) 保存本机原料、四份模型配置和四份 checkpoint 的相对路径、字节数及 SHA256。SHA256 是文件内容的指纹；服务器比对相同指纹，才能确认没有传错同名文件。索引只读文件生成，不代表权重已成功加载。若指纹不同，先查明来源和差异，不自动改基准、重算标签或换素材。

## 服务器配置

确认服务器没有同名本地配置后，从仓库根目录复制：

```bash
cp -n case_study/config.json case_study/config.server.local.json
```

在副本中设置以下字段，其余固定协议保持不变：

| JSON 字段 | 设置方式 |
|---|---|
| `paths.raw_root` | 服务器实际的 `rawdataset` 绝对路径，其下直接有 `Dry/`、`Noise/`、`RIR/` |
| `paths.output_root` | 如 `case_study/outputs/air_three_rooms_ch0_server_v1`；必须是 `case_study/outputs/` 下尚未使用的版本目录 |
| `evaluation.device` | 确认空闲 GPU 后使用 `cuda:0` 等进程可见索引，或明确 `cpu`；当前不支持 MPS |
| `evaluation.batch_size` | 默认 8；按可用显存调整，不改变样本设计；`num_workers` 先保持 0 |

所有相对路径按仓库根目录解析。不要改 `experiment_id`、三案例身份、历史标签、通道、裁剪、语音数量、SNR、噪声顺序和四组模型映射。数据目录与输出目录不能重叠。若设置 `CUDA_VISIBLE_DEVICES=2`，脚本中的 `cuda:0` 指所选物理 GPU 2；先以当前服务器的实际设备和占用为准。历史文档记录过 GPU 0/1 的 NVLink 问题，不代表当前硬件状态已验证。

四个 `best_model.pth` 及其 YAML 已被 Git 跟踪，路径由 `config.json` 固定；正常完整拉取应带齐，但服务器仍须核对真实文件、字节数和哈希，不自动寻找“最新”权重。当前脚本使用 `torch.load(weights_only=True)` 和严格 `state_dict` 匹配；服务器首次预检不调用它，能否实际加载仍待后续小批检查。

## 首次接手：环境与文件预检

优先复用原训练／测试环境。历史 Python 路径为 `/mnt/tidal-sh01/usr/chuan/youling/demucs_xxn/bin/python`，仅供定位，尚未在当前服务器确认。检查 Python 路径与版本、NumPy、SciPy、SoundFile/libsndfile、torch、配套 torchaudio、einops、PyYAML、音频解码后端，以及 GPU、驱动、空间和依赖冲突。不同 torchaudio 版本对 TorchCodec／FFmpeg 的要求不同，按实际版本核验；不直接安装最新版覆盖原环境。Case Study 不使用 PESQ、STOI 或 SwanLab。

可以读取包元数据、配置、音频头、标签 CSV，计算文件哈希，执行 `pip check`、查询 GPU 状态和空间。必要的依赖导入及 CUDA 可用性检查不等于加载 checkpoint。禁止用正常启动合成／评估“跑到报错”的方式预检：它们可能先创建输出目录再失败；也不调用 `torch.load`、模型前向、卷积合成或 T60 重算。

将结果另存 `case_study/outputs/preflight_<日期或版本>/REPORT.md`，列出实际 Python、原料根路径、提交、文件指纹比对、依赖与 GPU 状态、通过／失败／未验证项。只写该报告及服务器本地配置，不创建正式实验的 `data/` 或 `evaluation/`。音频纯采样载荷理论约 2.49 GB，此外要留模型、元数据及文件系统空间；这不是实际耗时或磁盘实测。

## 后续阶段及停止点

1. **现在：** 完成预检和本地路径配置，报告阻碍后停下；缺原料时列出精确路径，不重新下载不明版本凑齐。
2. **建议下一步：** 单独准备 36 条小批检查入口（1 干声 × 3 房间 × 2 噪声 × 6 SNR，四模型共 144 条预测）。它用于查流程错误，不用于决定哪组对照或挑选有利结果。当前没有该入口，不能把正式配置的 90／3240 校验改小来代替；准备和执行需以用户后续指令为准。
3. **小批通过且用户允许后：** 执行正式 3240 条合成，再对四组 checkpoint 评估。运行命令见 README 的“后续执行方式”。两阶段分别检查最终 `status: complete`、manifest、样本对齐及哈希；不静默丢样，不覆盖已有目录。失败保留残留，改用新的版本目录。

三案例固定为 office 0.454 s、lecture 0.773 s、stairway 1.138 s；标签是历史文件级标量，未证明是通道 0 专属真值。沿用 `unknown_file_level_historical`，不将索引 0 直接称为左耳。3240 个输入在四模型间完全配对，报告 RMSE、MAE、Bias；每个案例标签固定，不计算房间内 PCC／R²。三个场景的结果不能自动证明训练未见房间泛化、跨 seed 稳定、统计显著或去噪增益的机制。

## Codex CLI 启动

在服务器已有完整仓库且工作区无阻碍拉取的改动时：

```bash
cd /服务器项目路径/CMGAN-t60
git status --short --branch
git switch exp/kan-multitask
git pull --ff-only origin exp/kan-multitask
codex "请阅读 case_study/SERVER_PROMPT.md，按其中的范围接手当前任务。"
```

路径需替换；若服务器未克隆，先 `git clone --branch exp/kan-multitask https://github.com/xfyyye/CMGAN-t60.git`。遇到未提交改动、分支分叉或拉取冲突，保留现场并报告，不自动 reset、stash 或强推。可先激活已确认的原实验环境，再启动 Codex；尚不清楚环境时交给它按任务说明定位。

启动方式按 [Codex CLI 官方说明](https://learn.chatgpt.com/docs/codex/cli)核对；本机 `codex --help` 也确认交互入口接受初始提示词。服务器安装版本和登录状态尚未验证。启动命令只开启接手会话，任务范围由下面的独立提示词限定。
