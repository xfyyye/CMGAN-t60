# Prompt: 多任务 T60 + 去噪联合训练主架构图

## 使用说明

**上传图片（必须 2 张同时上传）：**
1. **主参考图**：`docs/figs/overall_architecture.png` —— 单任务原图，作为整体配色 / 字体 / 图例 / 编号风格的锚点
2. **风格参考图**：`docs/figs/refs/cmgan_fig2a_decoder_style.png` —— CMGAN 论文 Fig. 2a，作为 Mask/Complex Decoder 与 spectrogram 算子的画法锚点

**输出目标文件名建议**：`docs/figs/overall_multitask.png`

**期望迭代次数**：2~3 轮（第 1 轮看布局，第 2 轮修文字，第 3 轮调配色 / 对齐）

---

## Prompt 正文（复制以下全部内容到 ChatGPT）

```
# 角色 & 目标
你是一名 IEEE 期刊配图编辑师。我同时上传两张参考图：
  [图 1] 单任务 T60 估计架构图 (PNG, 我的原图)
  [图 2] CMGAN 论文 Fig. 2a (Encoder-decoders generator architecture)

请基于 [图 1] 的整体版式与配色, 参考 [图 2] 中 Mask Decoder /
Complex Decoder / spectrogram 算子的画法, 重画一张
「多任务版本: T60 估计 + 去噪辅助任务」的架构图。
最终用于 IEEE 期刊论文, 出版级清晰度、矢量级简洁观感。

# 风格锚定 (严格继承 [图 1])
- 沿用 [图 1] 全部色板:
    橙 = Conv,  粉 = InstanceNorm + PReLU,  青 = Dilated Conv,
    棕 = LayerNorm,  绿框 = FourierKAN (novel),  紫 = PReLU,
    蓝 = GLU,  橙菱形 = Swish,  红菱形 = Sigmoid,  灰圆 = Reshape
- 沿用 [图 1] 字体 (Latin Modern / Times 系列), 字号, 行高
- 沿用 [图 1] 细黑直角折线箭头 + 小箭头头
- 沿用 [图 1] 圆角矩形描边粗细 (~1pt)
- 沿用 [图 1] ①②③ 黑底白字圆形数字标号
- 白底, 无水印, 无装饰

# 关键: 从 [图 2] 复用的视觉元素
- Mask Decoder / Complex Decoder 的 DilatedDenseNet 必须画成
  [图 2] 那种「3D 倾斜砖块堆」, 而不是文字方块。
  砖块堆内部沿用 [图 1] 的色块（橙 Conv + 粉 IN+PReLU + 青 Dilated Conv）,
  外用虚线框圈住, 下方标 d = {1, 2, 4, 8}
- Sub-Pixel ConvT 必须画成 [图 2] 的「黄色实心圆 + 内嵌 ↑ 箭头」
  样式, 标 "Sub-Pixel", 不要画成矩形方块
- spectrogram (mask, 复频谱, 最终频谱) 用 [图 2] 的「灰阶频谱小图」
  样式, 不要用纯色矩形代替
- 掩码乘法用 [图 2] 的「圆圈内大写 M」算子
- 复数残差加法用 [图 2] 的「圆圈内 ⊕」算子
- ISTFT 块用 [图 2] 那个小紫色方框 "ISTFT" 样式

# 整体布局 (横向 16:10, 三分支结构)

整张图分为三个垂直区:
  (上) 共享前端: x(t) → STFT → Dense Encoder → TS-Conformer × N=2
  (中) 三分支分叉点 (从 TS-Conformer 输出引出, 标 shared features B×64×T×F/2)
  (下) 三个并行 decoder 分支, 自左向右:
       [Branch 1: T60 Head]   [Branch 2: Mask Decoder]   [Branch 3: Complex Decoder]
  (再下) Branch 2 和 Branch 3 的输出汇合, 经 spectral recombination → iSTFT → ŝ(t)
  (底) 损失公式 L_total = α · L_denoise + β · L_T60

# A. 共享前端 (从 [图 1] 直接保留, 不要重画)
逐元素保留 [图 1] 中:
  ① x(t) 波形图 + B×64000 标注
  ② STFT 块 (B×64000 → B×3×T×F)
  ③ Dense Encoder 完整 3D 砖块堆 (Conv 1×1 → IN+PReLU → 4 层 Dilated Conv
     d=1,2,4,8 → IN+PReLU → Conv Downsample) + B×64×T×F/2 输出标注
  ④ TS-Conformer × N (N=2) 紫色块 + B×64×T×F/2 输出标注

※ 重要: [图 1] 原右上角的「TS-Conformer Block 详图」(含 Conformer-Time /
Conformer-Freq / Conv Module) 全部删除。在 ④ 块右侧加一行小灰字
"(see Fig. 3 for TS-Conformer block details)"

# B. 分叉点
④ TS-Conformer 输出端引出一根主箭头, 标 "shared features  B×64×T×F/2"。
这根箭头分叉为三根, 分别向下指向三个分支的入口。
三个分支水平等距排列, T60 在最左、Mask 居中、Complex 在最右。

# C. Branch 1 — T60 Head (简化, 仅 4 个核心块)

T60 Branch 用浅紫色虚线外框, 标题 "T60 Branch"。内部自上而下:

  分叉箭头入口处, 在箭头旁标小灰字: "global pooling (μ ∥ max ∥ σ)   B×192"
  (不画 Mean Pool / Multi-Stat Pool 方块, 仅文字标注)

  ⑤  Linear + LayerNorm          (白底圆角矩形, 同 [图 1] 风格)
      下方维度: B × 64

  ⑥  FourierKAN Head              (绿色粗框, 标 "novel")
      框内: 嵌入一个小型 KAN 拓扑示意 ——
        左侧 4 个空心圆节点 (最右一个旁加 "···"),
        右侧 3 个空心圆节点 (同样加 "···"),
        全连接 12 条边, 每条边画成轻微下凹的贝塞尔曲线,
        曲线上叠加 1~1.5 个周期的小正弦波 (线宽 ~0.8pt),
        不同边用浅蓝/浅绿/浅橙轮换上色
      拓扑下方 10pt 灰字:
        "edges = learnable Fourier activations  {a_iω, b_iω}_{ω=1..Ω}"
      最底部超参标签: "Ω = 16 (Layer 1) / Ω = 8 (Layer 2)   ·   64 → 32 → 16"
      下方维度: B × 16

  ⑦  Linear + Sigmoid            (白底圆角矩形)
      下方维度: B × 1

  ⑧  Denorm                       (白底圆角矩形)

  → T̂60 (红色斜体输出标签, 同 [图 1] 风格)
  → 红色小标签: L_T60   (weight β)

# D. Branch 2 — Mask Decoder (CMGAN 3D 砖块风格)

Mask Decoder 用浅灰色虚线外框, 标题 "Mask Decoder"。
内部从上到下严格复刻 [图 2] 的 Mask Decoder 视觉:

  (1) DilatedDenseNet 砖块堆:
      4 块 dilated conv (青色 3D 砖块) + 首尾各夹 Conv+IN+PReLU
      (橙+粉砖块)
      虚线框圈住, 下方标 d = {1, 2, 4, 8}
      输出尺寸标: B × 64 × T × F/2

  (2) 黄色实心圆 + ↑ 箭头, 旁标 "Sub-Pixel"
      下方维度: B × 64 × T × F

  (3) 一个橙色砖块 (Conv 1×2) + 粉砖块 (IN+PReLU) + 橙砖块 (Conv 1×1)
      + 紫色三角 (PReLU, num_features = 201)

  (4) 输出: 一个小灰阶频谱图标, 标 "mask"
      尺寸: B × 1 × T × F

# E. Branch 3 — Complex Decoder (CMGAN 3D 砖块风格)

Complex Decoder 用浅灰色虚线外框, 标题 "Complex Decoder"。
内部从上到下:

  (1) DilatedDenseNet 砖块堆 (同 Branch 2 的画法)
      输出尺寸: B × 64 × T × F/2

  (2) 黄色实心圆 + ↑ 箭头, 旁标 "Sub-Pixel"
      输出: B × 64 × T × F

  (3) 一个粉砖块 (IN+PReLU) + 橙砖块 (Conv 1×2)

  (4) 输出: 一个小灰阶复频谱图标, 标 "Δ (real, imag)"
      尺寸: B × 2 × T × F

# F. Spectral Recombination (用算子 + 频谱图标, 不要写公式)

Branch 2 和 Branch 3 的输出在下方汇合:

  Mask 输出  ──→  圆圈 M (mask 乘 magnitude) ──→  小频谱图标 (out_mag)
                       ↑
                  |X_noisy| (从 STFT 引一根灰色细虚线过来, 标 "noisy magnitude")
                       │
                       └→ 与 noisy phase ∠X 结合 (圆圈 ⊕ phase)
                           ──→  小复频谱图标 (out_mag · e^(j∠X))

  Complex 输出 Δ(r,i) ─────────────────→  ⊕ (圆圈加号)
                                            ↑
                              out_mag·e^(j∠X) 引入

  ⊕ 输出 ──→  最终重建复频谱图标 X̂

※ 全部用 [图 2] 同款圆圈算子 + 灰阶频谱小图, 不要在图里写任何代数公式。
※ 算子之间的箭头标注若需要, 用 10pt 灰字简短标 "magnitude" / "phase" 即可。

# G. iSTFT + 输出波形

X̂ ──→ 紫色小方框 "iSTFT (n_fft=400, hop=100)"
     ──→ 蓝色波形小图标
     ──→ 标签: ŝ(t)   B × 64000
     ──→ 蓝色小标签: L_denoise   (weight α)
     ──→ 灰字附注: "enhanced (denoised, reverb-preserved)"

# H. 底部损失公式 (居中, 与正文同字体)

  L_total  =  α · L_denoise  +  β · L_T60         (best: α = 1.0, β = 0.1)

# I. 顶部图例 (在 [图 1] 原图例基础上新增条目)

保留 [图 1] 原图例的 10 个条目, 新增 3 个条目排在右侧:
  ⓐ Sub-Pixel (黄色实心圆 + ↑ 图标)
  ⓑ DilatedDenseNet (虚线框图标)
  ⓒ iSTFT (紫色小方框图标)

# 输出要求
- 横向 16:10 比例, PNG, 等效 ≥300 DPI
- 白底, 无水印, 无生成器标识
- 所有英文标签拼写须与本说明完全一致
- 整图视觉密度均衡, 三分支水平对齐, 留白充足
- 三分支视觉权重相近 (不要让某个分支明显过大或过小)
```

---

## 迭代提示 (常见问题排查)

| 翻车现象 | 第二轮怎么说 |
|---|---|
| 三分支顶部没对齐 | "三个 decoder branch (T60 / Mask / Complex) 的顶部入口必须水平对齐到同一条基准线" |
| Sub-Pixel 画成方块了 | "Sub-Pixel 必须用 [图 2] 中那种黄色实心圆 + ↑ 箭头的样式, 不要用矩形方块" |
| DilatedDenseNet 是平面方块 | "Mask/Complex Decoder 的 DilatedDenseNet 必须画成 [图 2] 那种 3D 倾斜砖块堆, 不能用平面矩形" |
| KAN 拓扑边曲线杂乱 | "FourierKAN 框内的节点拓扑边数控制在 12 条, 每条边曲线不要交叉, 用 3 种浅色轮换" |
| 公式被画进图里了 | "Spectral Recombination 区域不要写任何代数公式, 只画 ⓜ 圆圈和 ⊕ 圆圈算子" |
| 输出底色不是白色 | "整张图背景必须是纯白 #FFFFFF, 不要加任何阴影或渐变" |
| 文字标签拼错 | 截屏标红错处发回 "把红框内的 'XXX' 改为 'YYY', 其他不要动" |

## 兜底方案

如果 FourierKAN 内部的 KAN 拓扑画了 2 轮还是不满意, 让 ChatGPT 在 FourierKAN 绿框内**留一块空白浅米白矩形**, 只画好框 + 超参标签 `Ω = 16 (Layer 1) / Ω = 8 (Layer 2)`。拓扑示意图用 Keynote / Figma 五分钟手画后合成进去。
