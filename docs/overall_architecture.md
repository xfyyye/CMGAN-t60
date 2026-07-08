# 角色 & 目标
  你是一名学术论文配图编辑师。我上传的 PNG 是一张语音模型
  (CMGAN + FourierKAN) 的单任务 T60 估计架构图。请在严格继承
  原图风格的前提下，把它改造为"多任务版本: T60 估计 + 去噪辅助任务"
  的架构图。最终用于 IEEE 期刊论文，要求出版级清晰度、矢量级简洁观感。

  # 风格约束 (务必严格继承原图，不要引入任何新色调或新字体)
  - 沿用原图色板：橙=Conv、粉=InstanceNorm+PReLU、青=Dilated Conv、
    棕=LayerNorm、绿框=FourierKAN (novel)、紫=PReLU、蓝=GLU、
    橙菱形=Swish、红菱形=Sigmoid、灰圆=Reshape
  - 同款字体 (Latin Modern / Times 系列)、同款字号、同款行高
  - 同款细黑直角折线箭头 + 小箭头头
  - 同款圆角矩形描边粗细 (~1pt)
  - 同款 ①②③ 黑底白字圆形数字标号
  - 白底、无水印、无装饰元素

  # A. 必须逐像素保留 (不要重画)
  1. 左侧主流程 ①~④：x(t) → STFT → Dense Encoder → TS-Conformer×N=2
     及其所有尺寸标注 (B×64000, B×3×T×F, B×64×T×F/2 等)
  2. 右上角「TS-Conformer Block」详图框
     (Conformer-Time / Conformer-Freq / Conv Module 三部分及全部内部标签)
  3. 左下原 ⑤~⑩ 整条 T60 估计链路 (Mean Pool → Multi-Stat Pool →
     Linear+LayerNorm → FourierKAN Head → Linear+Sigmoid → Denorm → T̂60)
     模块、位置、张量形状全部不动；这条链路在新图里改称为「T60 Branch」
  4. 顶部图例 (Conv. / InstanceNorm+PReLU / Dilated Conv. / LayerNorm /
     FourierKAN Layer / Reshape / PReLU / GLU / Swish / Sigmoid) 全部保留

  # B. 删除并替换
  1. 右下「Fourier-KAN Regression Head」详图框内两个 Σ 求和公式
     (y_j = ΣΣ[a·cos + b·sin] + bias_j) 彻底删除
  2. 替换为「KAN 拓扑示意」：每个 FourierKAN Layer 绿框内画一个小拓扑
     - Layer 1 框 (Ω=16, 64→32): 左侧 4 个空心圆节点 (最右一个旁加 "···"),
       右侧 3 个空心圆节点 (同样加 "···"), 全连接 12 条边；
       每条边画成轻微下凹的贝塞尔曲线，曲线上叠加 1~1.5 个周期的
       小正弦波 (线宽 ~0.8pt)，不同边用浅蓝/浅绿/浅橙轮换上色
     - Layer 2 框 (Ω=8, 32→16): 3 个输入点 → 2 个输出点, 6 条边,
       正弦周期数比 Layer 1 少一半，色调略冷
     - 拓扑下方一行 10pt 灰字:
       "edges = learnable Fourier activations  {a_{i,ω}, b_{i,ω}}_{ω=1..Ω}"
     ※ 兜底: 如果上述节点+正弦曲线过于复杂导致渲染杂乱,
       请在两个绿框内各留一块浅米白空白矩形 (供我后期手动嵌入),
       只画好绿框 + 超参标签 "Ω = 16  ·  64 → 32" / "Ω = 8  ·  32 → 16" 即可
  3. 右侧原 "Hierarchical Ω design ..." 文字说明保留, 对齐到新拓扑右侧

  # C. 新增「Denoise Branch (Auxiliary Task)」—— 放在主流程右侧并行下行
  从 ④ TS-Conformer×N=2 输出端引一根分叉箭头, 标注
  "shared features  B×64×T×F/2", 分别指向左下已有的 T60 Branch
  和右下新增的 Denoise Branch。Denoise Branch 用一个浅灰底外框包起来,
  标题 "Denoise Branch (Auxiliary Task)"。内部自上而下:

    ⓐ 两条并列子链 (左右排列)

        —— Mask Decoder ————          —— Complex Decoder ——
        DilatedDenseNet (depth=4)     DilatedDenseNet (depth=4)
                ↓                              ↓
        SubPixel ConvT (1×3, ×2)      SubPixel ConvT (1×3, ×2)
                ↓                              ↓
        Conv 1×2                      InstanceNorm + PReLU
                ↓                              ↓
        InstanceNorm + PReLU          Conv 1×2
                ↓                              ↓
        Conv 1×1                      Δ (real, imag)
                ↓                       B × 2 × T × F
        PReLU (num_features=201)
                ↓
        mask    B × 1 × T × F

    ⓑ 一个浅紫色圆角矩形汇合块 "Spectral Recombination", 内三行小字:
          out_mag = mask · |X_noisy|
          real = out_mag · cos(∠X) + Δ_real
          imag = out_mag · sin(∠X) + Δ_imag

    ⓒ "iSTFT (n_fft=400, hop=100)" 方框 (与原图 STFT 同款配色)

    ⓓ 输出波形小图标 ŝ(t), 标签
      "enhanced (denoised, reverb-preserved)   B × 64000"

  # D. 损失标签 (这是"多任务"卖点, 必须画上)
  - T60 Branch 末端 T̂60 旁加红色小标签:    L_T60      (weight β)
  - Denoise Branch 末端 ŝ(t) 旁加蓝色小标签: L_denoise  (weight α)
  - 整张图最底部居中一行公式 (与正文字体一致):
      L_total = α · L_denoise + β · L_T60     (best: α = 1.0, β = 0.1)

  # E. 图例新增三个条目 (与原图右上角图例同款样式)
  - DilatedDenseNet  (深蓝方块)
  - SubPixel ConvT   (橙色方块)
  - iSTFT            (深灰方块)

  # 输出要求
  - 横向 16:10 或 A4 横版比例 (容纳新增右侧 Denoise Branch)
  - 白底, PNG, 等效 ≥300 DPI
  - 不添加任何水印、署名、生成器标识
  - 所有英文标签拼写须与本说明完全一致