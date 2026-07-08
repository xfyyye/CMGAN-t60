# Prompt: TS-Conformer Block 详图 (独立子图)

## 使用说明

**上传图片（仅 1 张）：**
1. **唯一参考图**：`docs/figs/overall_architecture.png` —— 单任务原图，TS-Conformer Block 的详图就在它的右上角，作为这一版独立子图的内容来源 + 风格锚点

**输出目标文件名建议**：`docs/figs/tsconformer_block.png`

**期望迭代次数**：1~2 轮（这张图内容相对单一，AI 易收敛）

**论文里如何引用**：作为主架构图（`overall_multitask.png`）的伴生子图，标号建议 Fig. 3，caption 类似：
> "Detailed structure of the TS-Conformer block used in Fig. 2."

---

## Prompt 正文（复制以下全部内容到 ChatGPT）

```
# 角色 & 目标
你是一名 IEEE 期刊配图编辑师。我上传的 PNG 是一张语音模型的
单任务架构图, 其右上角包含一个「TS-Conformer Block」详图。
请将这个 TS-Conformer Block 详图独立提取出来, 重新排版为一张
横向独立子图 (standalone figure), 用于论文中作为主架构图的伴生
子图 (类似 CMGAN 论文 Fig. 2b 的角色)。

注意: 我不需要原图的其他任何部分 (波形 / STFT / Dense Encoder /
T60 head 等都不要), 只要 TS-Conformer Block 这一个详图重新展开。

# 风格锚定 (严格继承原图)
- 沿用原图色板:
    橙 = Conv,  粉 = InstanceNorm + PReLU,  青 = Dilated Conv,
    棕 = LayerNorm,  紫 = PReLU,  蓝 = GLU,
    橙菱形 = Swish,  红菱形 = Sigmoid,  灰圆 = Reshape,
    黄 = FFN,  绿 = MHSA,  红 = Conv Module (沿用原图右上区那套配色)
- 沿用原图字体 (Latin Modern / Times 系列), 字号比原图大约 1.2 倍
  (因为现在是独立图, 有更多空间放大)
- 沿用原图细黑直角折线箭头 + 小箭头头
- 沿用原图圆角矩形描边粗细 (~1pt)
- 白底, 无水印, 无装饰

# 整体布局 (横向, 三块从上到下)

整图分为三个水平排列的子模块, 自上而下垂直堆叠:

  ┌──────────────────────────────────────────────────┐
  │             Conformer - Time                      │   (子模块 1)
  └──────────────────────────────────────────────────┘
                          │
  ┌──────────────────────────────────────────────────┐
  │             Conformer - Freq                      │   (子模块 2)
  └──────────────────────────────────────────────────┘
                          │
  ┌──────────────────────────────────────────────────┐
  │       Conv Module  (k = 31, depthwise)            │   (子模块 3)
  └──────────────────────────────────────────────────┘

子模块 1 和子模块 2 是 Conformer Block 的两个时间/频率方向变体,
子模块 3 是 Conformer Block 内部用到的 Conv Module 展开详图,
用一根虚线从子模块 1 或 2 中的 "Conv Module" 块引下来指向子模块 3。

# A. Conformer-Time (子模块 1)

水平流水线, 从左到右依次:

  ┌─────────┐  ┌────┐   ┌──────────────┐   ┌────────────┐   ┌────┐   ┌─────────┐  ┌─────────┐
  │ Reshape │  │FFN │   │     MHSA     │   │ Conv Module│   │FFN │   │LayerNorm│  │ Reshape │
  │BF'×T×C  │→ │×0.5│→⊕→│  (4 heads,   │→⊕→│  (k=31,    │→⊕→│×0.5│→⊕→│         │→ │B×T×F'×C │
  └─────────┘  └────┘   │   Shaw RPE)  │   │  depthwise)│   └────┘   └─────────┘  └─────────┘
                        └──────────────┘   └────────────┘

- 每个 ⊕ 是圆圈加号 (residual connection)
- 每条残差路径用虚线从 ⊕ 之前引到 ⊕ 之上, 表示 skip-connection
- 头部和尾部的 Reshape 用灰色圆圈, 内部模块用对应配色的圆角矩形
- 子模块 1 框外左上角小灰字标题: "Conformer-Time"

# B. Conformer-Freq (子模块 2)

视觉与子模块 1 完全对称, 仅两处不同:
- 入口 Reshape 标签改为: BT × F' × C
- 出口 Reshape 标签改为: B × T × F' × C  (同上)
- 子模块 2 框外左上角小灰字标题: "Conformer-Freq"

# C. Conv Module 展开 (子模块 3)

水平流水线, 从左到右依次:

  ┌────────────┐   ┌─────────────┐   ┌─────┐   ┌──────┐   ┌────────────┐   ┌──────┐   ┌────────┐
  │ Pointwise  │   │  Depthwise  │   │ GLU │   │Swish │   │ Pointwise  │   │ PReLU│   │Dropout │
  │ Conv (1×1) │ → │ Conv (k=31) │ → │     │ → │      │ → │ Conv (1×1) │ → │      │ → │        │
  └────────────┘   └─────────────┘   └─────┘   └──────┘   └────────────┘   └──────┘   └────────┘

- Pointwise Conv 用浅橙色圆角矩形
- Depthwise Conv 用青色圆角矩形
- GLU 用蓝色菱形
- Swish 用橙色菱形
- PReLU 用紫色倒三角
- Dropout 用浅灰色圆角矩形
- 子模块 3 整体用一个浅紫色虚线外框包住, 框外左上角标题:
  "Conv Module  (k = 31, depthwise)"
- 从子模块 1 或子模块 2 中的 "Conv Module" 块引一根虚线下来,
  指向子模块 3 的左上角, 表示「展开详图」关系

# D. 顶部图例 (新增独立图例栏)

由于是独立子图, 必须自带图例。在图最顶部水平排列:
  Conv. (橙)  ·  InstanceNorm + PReLU (粉)  ·  LayerNorm (棕)  ·
  Reshape (灰圆)  ·  PReLU (紫三角)  ·  GLU (蓝菱形)  ·
  Swish (橙菱形)  ·  FFN (黄)  ·  MHSA (绿)  ·
  Dilated Conv. (青)

每条图例同原图样式: 小色块 + 文字标签, 横向排列, 间距均匀。

# 输出要求
- 横向 16:9 或 4:3 比例 (容纳上下三个子模块)
- 白底, PNG, 等效 ≥300 DPI
- 不添加任何水印、署名、生成器标识
- 三个子模块之间留白充足 (子模块间距 ≥ 子模块本身高度的 30%)
- 所有英文标签拼写须与本说明完全一致
- 比原图右上角那个详图视觉上更宽松、更易读
```

---

## 迭代提示 (常见问题排查)

| 翻车现象 | 第二轮怎么说 |
|---|---|
| 子模块 1 和子模块 2 视觉不对称 | "Conformer-Time 和 Conformer-Freq 必须视觉完全对称, 仅入口/出口 Reshape 的张量标签不同" |
| Conv Module 那条虚线引线没画 | "在 Conformer-Time 的 Conv Module 块右下角引一根细虚线斜向下, 指向最底部 Conv Module 展开图的左上角" |
| 残差 ⊕ 圆圈缺失 | "每个 sub-layer (FFN, MHSA, Conv Module, FFN) 之后都必须有一个圆圈 ⊕, 并从该 sub-layer 之前引一根虚线弧绕到 ⊕ 之上作为 skip-connection" |
| FFN ×0.5 标记没了 | "首尾两个 FFN 块内必须显示 '×0.5' 字样, 表示 half-step feed-forward" |
| 头尾 Reshape 张量标签错 | "Conformer-Time 入口 BF'×T×C, 出口 B×T×F'×C; Conformer-Freq 入口 BT×F'×C, 出口 B×T×F'×C" |
| 图例排不下 | "顶部图例如果一行排不下, 排成两行均匀分布, 字号不要缩小" |

## 备选简化方案

如果你觉得三个子模块挤在一张图太密, 可以让 ChatGPT 只画 **Conformer-Time + Conv Module 展开**两块 (因为 Conformer-Freq 视觉上完全对称, 在论文 caption 里加一句 "Conformer-Freq is identical except for the reshape dimensions, see Sec. X" 即可)。
