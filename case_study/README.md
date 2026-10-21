# 三个 AIR 房间的独立 Case Study

本目录只包含新增的合成、评估和配置代码，不修改或调用原训练入口。2026-10-21 本轮仅准备代码并做静态检查；**没有运行合成/评估脚本、加载模型权重或生成音频及预测，也没有做运行时测试。**

**服务器接手先读 [SERVER_HANDOVER.md](SERVER_HANDOVER.md)，交给 Codex CLI 的任务说明在 [SERVER_PROMPT.md](SERVER_PROMPT.md)。** `case_study/` 是后续 Case Study 的独立子项目；终端和 Codex 的工作目录应是整个 `CMGAN-t60` 仓库根目录。原始数据放在仓库外，服务器路径写入不提交的 `config.server.local.json`；不要把原料拷入本目录。源文件大小和 SHA256 基准见 [ASSET_INDEX.json](ASSET_INDEX.json)。

## 固定参数

| 参数 | 值 |
|---|---|
| 配置入口 | `case_study/config.json`；相对路径均按仓库根目录解析 |
| RIR／历史标签 | office `T60_0.454_000.wav`／0.454 s；lecture `T60_0.773.wav`／0.773 s；stairway `T60_1.138.wav`／1.138 s |
| RIR通道 | 固定索引0，不做通道波形平均、起点截断、尾部裁剪或T60整形 |
| 干声 | `Dry/TIMIT_90_test_dry`中全部90个排序WAV；16 kHz、单声道、128000点 |
| 卷积／观察窗 | 完整8 s干声线性卷积后取`[32000:96000]`，输出64000点／4 s |
| 噪声 | `white.wav`、`babble.wav`，各取4 s；每个干声×噪声固定一个窗口 |
| SNR | −5、0、5、10、15、20 dB；定义在最终4 s无噪混响语音与噪声的全段功率上 |
| 随机性 | seed42；排序干声×配置噪声顺序预抽180个窗口，跨三个房间及六档SNR复用 |
| 保存 | 三路单声道FLOAT WAV：含噪、无噪混响、缩放后噪声；共同增益`min(1,0.99/三路最大峰值)` |
| 写后核查 | 16 kHz、单声道、64000点、有限值；SNR偏差≤0.01 dB，加性误差≤1e−6 |
| 数量 | 3×90×2×6＝3240个输入样本，每房间1080；三路合计9720个WAV |
| 评估 | 内部四组、batch8、workers0、CPU/CUDA auto、FP32，无autocast/TF32；只报告RMSE、MAE、Bias |

RIR和噪声路径相对`paths.raw_root`；默认原料位置为`/Users/xiexinni/dataset/rawdataset`。上服务器时可修改`raw_root`。输出必须位于仓库`case_study/outputs/`下的一个独立版本目录，不能指向旧`runs`或原料。

固定设计的数值、三个RIR身份和四个模型映射由`common.py`校验。遇到缺文件、无效能量、异常长度、标签来源不符或数量不完整时终止，不自动换语音、换RIR、丢样或用随机模型代替。`min_power=1e-12`是数值有效性下限，不是语音活跃度检测；未实现按模型表现筛样本。

## 四组冻结模型

| ID | 设置 | 模型配置 | checkpoint |
|---|---|---|---|
| `dact_kan_multitask` | KAN，多任务 | `configs/t60_multitask/kan_mse_norm_w_denoise_b005_seed42.yaml` | `runs/kan_multitask_norm_w_denoise_b005_seed42/best_model.pth` |
| `matched_mlp_multitask` | 匹配MLP，多任务 | `configs/t60_multitask/mlp_matched_drop01_mse_norm_w_denoise_b005_seed42.yaml` | `runs/mlp_matched_drop01_multitask_norm_w_denoise_b005_seed42/best_model.pth` |
| `kan_single` | KAN，单任务 | `configs/t60_single/kan_v2_mse_seed42.yaml` | `runs/single_t60_kan_v2_mse_seed42/best_model.pth` |
| `matched_mlp_single` | 匹配MLP，单任务 | `configs/t60_single/mlp_matched_drop01_mse_seed42.yaml` | `runs/single_t60_mlp_matched_drop01_mse_seed42/best_model.pth` |

评估直接复用原`dataset.create_test_loader`和`utils.power_compress`：含噪能量归一化、Hamming窗、FFT400、hop100、0.3次方压缩恰一次，再按0.1–1.5 s线性反归一化。单任务读取张量输出，多任务取第3项。MLP为`[192,192,64]`、dropout0.1，KAN核对有效Fourier dropout0.1，不误用单任务配置中无效的通用dropout0.3。

四组配置和checkpoint文件已在之前的静态资产核查中确认存在，但本轮没有测试加载。脚本只接受严格匹配的state_dict；使用`weights_only=True`读取，若历史格式或依赖不兼容会明确报错，不自动转换或重新训练。外部三种方法未加入本目录。

## 后续执行方式（本轮未执行）

下面是**用户后续明确允许执行全量任务后**使用的命令，不是服务器首次接手时要执行的任务。首次接手先按 `SERVER_PROMPT.md` 做环境和文件预检；当前没有 `--smoke`、`--limit` 或 `--preflight` 入口，正式入口固定要求3240条，不能把运行至报错当预检。

使用与原实验兼容的Python≥3.9环境；需要NumPy、SciPy、SoundFile、PyTorch、配套torchaudio、einops、PyYAML，以及该torchaudio版本可用的音频解码后端。依赖参照仓库`requirements.txt`，本轮没有安装或检查服务器运行环境。不应直接升级原训练环境以匹配最新依赖。

从仓库根目录执行时，先合成：

```bash
python -m case_study.synthesize --config case_study/config.server.local.json
```

只有`data/synthesis_summary.json`最后写入`status: complete`，才算合成完成。随后评估：

```bash
python -m case_study.evaluate --config case_study/config.server.local.json
```

这两条命令相互独立，没有自动串联或后台任务。更换评估设备或batch可修改配置的`evaluation`部分，数据契约哈希不包含这些运行设置，能够复用同一份音频；完整评估配置仍另存快照和哈希。改动声道、语音/噪声选择、裁剪、SNR或案例标签则不能复用旧数据契约。

## 输出、来源和完成状态

默认输出根：`case_study/outputs/air_three_rooms_ch0_v1/`，已由本目录`.gitignore`排除实验大文件。

```text
air_three_rooms_ch0_v1/
  data/
    config.snapshot.json
    audio/<case_id>/<sample_id>/
      <sample_id>.wav
      <sample_id>_denoised.wav
      <sample_id>_noise.wav
    manifest.jsonl
    synthesis_summary.json
  evaluation/
    config.snapshot.json
    <model_id>.predictions.jsonl
    <model_id>.summary.json
    summary.json
```

音频`sample_id`最后两段仍是T60和SNR，兼容正式Dataset；manifest显式记录room、干声、噪声、原始/现存RIR、通道、标签CSV行、原料与输出SHA256、观察窗、噪声偏移、两级增益、实际SNR和保存后误差。合成汇总保存源文件/代码哈希、版本、RNG类型和全部180个偏移。

评估先验证合成完成标记、数据契约哈希和manifest哈希，再检查3240条完整配对设计、每份输入哈希与实际Dataset样本顺序。模型间、评估前后都会核对输入；结果保存未舍入的逐样本预测及五层汇总：整体、房间、房间×噪声、房间×SNR、房间×噪声×SNR。每个房间参考值固定，因此不计算PCC/R²。

`data`或`evaluation`目录只要已存在，即拒绝覆盖。失败时可能留下部分输出，但不会有最终完成标记；如需另一次运行，应选择新的版本输出目录，不清理或覆写旧结果。3240组三路FLOAT音频的纯采样载荷理论约2.49 GB，另有文件头、manifest和预测记录；这不是实测磁盘/耗时结果。

## 标签和解释边界

标签沿用历史RIR文件标量，`label_channel_policy=unknown_file_level_historical`；记录了CSV映射，但未证明标签就是通道0真值。索引0不直接称左耳；本脚本不重算T60，也不做已取消的表征分析或梯度探针。

三个RIR不等于大量独立房间；重复语音、噪声窗口和SNR是配对条件，12960条模型预测不是12960个独立样本。输出用于三个已选场景下相对历史参考标签的诊断，不自动构成训练未见房间泛化、机制或统计显著性证据。
