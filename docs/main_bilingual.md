# 中英段落对照审阅文档

> 每段中文后紧跟英文，逐段审阅英文书写。

---

# ═══ 摘要 / Abstract ═══

**🇨🇳** 混响时间 $T_{60}$ 是最具信息量的房间声学参数之一，然而从单通道含噪混响录音中进行盲估计仍然困难，因为晚期混响尾在能量上天然偏弱，极易被加性噪声掩蔽。我们提出一个面向盲 $T_{60}$ 估计的多任务学习框架：以 $T_{60}$ 回归为主任务，以去噪重建（含噪混响语音 $\to$ 无噪混响语音）为辅助任务，二者共享一个 dual-path time–frequency Conformer backbone。含噪混响语音的复短时傅里叶频谱经稠密卷积编码器与两个 time–frequency Conformer 块（TSCB）处理后，瓶颈特征由多统计量池化层聚合，再经两层 Fourier-KAN 头映射到 $T_{60}$；与此同时，同一瓶颈特征经掩码解码器与复数解码器重建增强语音。去噪辅助任务信息密集的逐时频点监督，迫使共享层学习精细的时频判别特征；我们验证了这种由去噪驱动的表征反过来会显著改善 $T_{60}$ 估计。与相近容量的多层感知机（MLP）回归头相比，Fourier 基把每条边上的激活函数表达为可学习的三角级数，与池化声学统计量到目标 $T_{60}$ 之间光滑但非线性的依赖关系天然契合。在覆盖仿真与真实房间脉冲响应（RIR）、已见与未见噪声类型的四个含噪混响测试集上，所提模型取得平均均方根误差（RMSE）92.91 ms、平均绝对误差（MAE）56.68 ms、Pearson 相关系数 0.9625；相比单任务消融基线（861,793 参数，RMSE 111.15 ms）RMSE 降低 16.4%。encoder 与 Conformer 块为两任务共享，去噪辅助任务新增两个解码器约 0.54M 参数，总参数 1.41M。梯度探针分析表明，两任务梯度在整个训练过程中近似正交（无系统性冲突），故可联合优化而互不干扰；任务权重消融进一步表明，去噪任务权重越大，$T_{60}$ 精度越高，而表征分析（CKA、线性探针）显示这一增益源于去噪任务对共享特征的**重塑**，而非任何梯度幅度主导效应。

---

**🇨🇳** **关键词：** 盲混响时间估计、多任务学习、去噪辅助监督、Conformer、Kolmogorov–Arnold 网络、Fourier 基、噪声鲁棒性

---

**🇨🇳** ---

---


# ═══ 引言 / Introduction ═══

**🇨🇳** 当声源在一个封闭空间中发声时，接收端不仅捕获直达声，还会接收到来自周围壁面的大量反射声（Kuttruff 等，2013）。这些反射声随时间衰减，共同构成房间的混响。声源停止后，声能衰减 60 dB 所需的时间被定义为混响时间 $T_{60}$（ISO 3382-2，2008）。作为一个概括房间能量衰减行为的标量，$T_{60}$ 是表征封闭空间声学属性最常用的参数之一，直接影响语音质量、可懂度，以及去混响、语音识别、声源定位、虚拟/增强现实渲染等下游音频应用的性能（Benzeghiba 等，2007；Anthes 等，2016）。标准化测量 $T_{60}$ 的方法包括中断噪声法与积分脉冲响应法（Schroeder，1965），二者均需要全向声源、校准过的传声器与训练有素的操作人员。但这类现场测量既不便携，且在大规模或现场测量时成本高昂、实现复杂。因此，*盲*估计方法应运而生：在没有任何专用激励信号或房间几何先验的前提下，直接从录制的语音信号中推断 $T_{60}$（Eaton 等，2016）。

*🇬🇧* When a sound is emitted in an enclosure, the receiver captures not only the direct path but also a dense set of reflections from the surrounding boundaries~[kuttruff2013room]. These reflections decay over time and together form the room reverberation. The time required for the sound energy to decay by 60\,dB after the source has been switched off is defined as the reverberation time $T_{60}$~[iso3382_2_2008]. As a single scalar that summarises the energy-decay behaviour of a room, $T_{60}$ is one of the most widely used parameters to characterise the acoustic property of an enclosure and has a direct impact on speech quality, intelligibility, and the performance of downstream audio applications such as dereverberation, speech recognition, sound source localisation, and virtual or augmented reality rendering~[benzeghiba2007asrreview, anthes2016vr]. The standardised ways of measuring $T_{60}$ are the interrupted-noise method and the integrated impulse response method~[schroeder1965rt], both of which require omnidirectional sources, calibrated microphones and trained operators. Such on-site measurements are nevertheless not portable, and are costly and complex to implement whenever large-scale or in-situ measurement is needed. Blind estimation methods have therefore emerged, in which $T_{60}$ is inferred directly from a recorded speech signal without any dedicated excitation or knowledge of the room geometry~[eaton2016ace].

---

**🇨🇳** 过去二十年间，盲 $T_{60}$ 估计积累了大量工作。传统方法依赖统计信号处理：基于模型的估计器将房间脉冲响应（RIR）的晚期部分建模为阻尼高斯过程（Polack，1988），从能量包络的衰减速率恢复 $T_{60}$；基于特征映射的估计器则从手工设计的声学描述子（如衰减速率的负侧方差、谐波分量的音高强度或谱衰减分布）中推导 $T_{60}$（Ratnam 等，2003；Wen 等，2008；Eaton 等，2013；Falk 等，2010；Löllmann 等，2010）。ACE 挑战赛对这些方法作了系统评测（Eaton 等，2016），同时也暴露出两个反复出现的困难：（i）多数估计器受加性噪声正偏置影响，因为类噪声的混响尾被埋没在噪声之中；（ii）解析衰减模型在非平稳或低信噪比（SNR）场景下十分脆弱。即便是带噪声鲁棒性改进的版本，如降低计算量的谱衰减分布估计器（Eaton 等，2013）或子带分解估计器（Prego 等，2015），仍需要一个相对准确的噪声水平先验估计，而噪声非平稳时这本身就是一个难题。

*🇬🇧* Over the past two decades, a large body of work on blind $T_{60}$ estimation has accumulated. The traditional family relies on statistical signal processing: model-based estimators describe the late part of the room impulse response (RIR) as a damped Gaussian process~[polack1988these] and recover $T_{60}$ from the decay rate of the energy envelope, while feature-mapping based estimators derive $T_{60}$ from a hand-designed acoustic descriptor such as the negative-side variance of the decay rate, the pitch strength of harmonic components or the spectral decay distribution~\citep{ratnam2003blind, wen2008negativeside, eaton2013sdd, falk2010temporal, lollmann2010iwaenc}. These methods have been thoroughly benchmarked by the ACE challenge~[eaton2016ace], which also revealed two recurring difficulties: (i) most estimators are positively biased by additive noise because the noise-like reverberation tail is buried in the noise, and (ii) the analytic decay model is fragile in non-stationary or low-SNR scenarios. Even noise-robust extensions such as the spectral-decay-distribution estimator with reduced computational cost~[eaton2013sdd] or the subband-decomposition based estimator~[prego2015subband] still require a relatively accurate prior estimate of the noise level, which is itself a hard problem when the noise is non-stationary.

---

**🇨🇳** 深度学习随后成为一种数据驱动的替代方案。Gamper 等（2018）将 log-mel 特征送入一个六层卷积网络，赢得 ACE 单通道赛道；Xiong 等（2018）将 $T_{60}$ 与早期混响比的联合估计建模为时频特征上的分类问题；Deng 等（2020）引入卷积循环网络以在变长语音上在线估计；Srivastava 等（2021）将该框架推广到多通道输入；Götz 等（2023）用三种卷积循环变体处理动态声学条件下的参数估计；Saini 等（2023）证明 MobileViT 风格的 transformer 可在智能手机上实时运行，而 Duangpummet 等（2022）则从全频带 $T_{60}$ 走向 $T_{60}$、早期衰减时间、清晰度与语音传输指数的联合估计。与本文最相关的是 Zheng 等（2022）和 Zhang 等（2024）：他们探索了将 $T_{60}$ 回归与一个辅助去噪目标配对的多任务形式，并观察到去噪任务能改善 $T_{60}$ 估计。尽管上述数据驱动模型在干净或高 SNR 条件下明显优于统计信号处理基线，两个普遍挑战仍贯穿这一系列工作。其一，晚期混响尾在能量上天然偏弱，易被真实噪声掩蔽，估计精度在低 SNR 或未见噪声场景下仍明显下降；已有工作主要依赖 backbone 自身容量来获取噪声鲁棒性，但这能否驱动底层学到精细的时频判别特征并不明朗，而精确的 $T_{60}$ 恰恰依赖对衰减包络细节的感知。其二，几乎所有现有估计器都在时频 backbone 之上堆叠全连接层作为回归器，每个神经元仅施加单一固定非线性，对池化瓶颈特征到标量 $T_{60}$ 这条光滑但非平凡的映射，表达能力有限。

*🇬🇧* Deep learning (DL) has subsequently emerged as a data-driven alternative. [gamper2018cnn] fed log-mel features to a six-layer convolutional neural network and won the ACE single-channel track; [xiong2018ercnn] posed the joint $T_{60}$ and direct-to-reverberant ratio estimation as a classification problem on time--frequency features; [deng2020crnn] introduced a convolutional recurrent network for online estimation on variable-length utterances; [srivastava2021blindroom] extended the formulation to multi-channel inputs; [goetz2023online] tackled parameter estimation under dynamic acoustic conditions with three convolutional recurrent variants; [saini2023mobileaudiotransformer] have demonstrated that a MobileViT-style transformer can run on a smartphone in real time, while [duangpummet2022stochastic] have moved from full-band $T_{60}$ towards a joint estimation of $T_{60}$, early-decay time, clarity and speech transmission index. Most relevant to this work, [zheng2022noiseaware] and [zhang2024damtl] explored multi-task formulations that pair $T_{60}$ regression with an auxiliary denoising objective and observed that the denoising task improves $T_{60}$ estimation. Although the data-driven models above clearly outperform the statistical signal processing baselines in clean or high-SNR conditions, two general challenges persist across this body of work. First, the late reverberation tail is intrinsically energy-weak and is easily masked by real-world noise, so estimation accuracy still degrades noticeably in low-SNR or unseen-noise scenarios; existing estimators mainly rely on the backbone's own capacity for noise robustness, yet it is unclear whether this alone drives the lower layers to learn the fine-grained time--frequency discriminative features on which an accurate $T_{60}$---itself dependent on the detailed shape of the decay envelope---ultimately relies. Second, almost all existing estimators place a stack of fully connected layers on top of the temporal/spectral backbone as the regressor, which applies a single fixed non-linearity per neuron and offers limited expressive power for the smooth but non-trivial mapping from pooled bottleneck statistics to a scalar $T_{60}$.

---

**🇨🇳** 为应对上述挑战，我们提出一个面向盲 $T_{60}$ 估计的多任务学习框架：以 $T_{60}$ 回归为主任务，同时联合训练一个去噪重建任务作为辅助，二者共享同一个 time–frequency Conformer backbone。这样做的动机有二。其一，$T_{60}$ 取决于混响衰减包络，而该包络集中在能量微弱的晚期混响尾中，极易被加性噪声掩蔽；引入去噪任务可迫使共享层学会从含噪频谱中分离噪声，使混响衰减结构得以暴露，进而惠及 $T_{60}$ 估计。其二，池化声学统计量到标量 $T_{60}$ 的映射光滑但非平凡，需要比常规 MLP 具有更强归纳偏置的回归头。为理解去噪辅助监督为何及如何起作用，我们进一步开展系统的梯度探针与表征分析。

*🇬🇧* To address these challenges, we propose a multi-task learning framework for blind $T_{60}$ estimation that pairs $T_{60}$ regression with a denoising reconstruction auxiliary task over a shared time--frequency Conformer backbone. The motivation is twofold. First, $T_{60}$ depends on the reverberation decay envelope, which resides mainly in the energy-weak late reverberation tail that is easily masked by additive noise; introducing a denoising task forces the shared layers to learn to separate noise from the reverberant spectrum, thereby exposing the decay structure and benefiting $T_{60}$ estimation. Second, the mapping from pooled acoustic statistics to a scalar $T_{60}$ is smooth but non-trivial, which calls for a regression head with stronger inductive bias than a conventional MLP. To understand why and how the denoising auxiliary supervision takes effect, we further conduct systematic gradient-probe and representation analyses.

---

**🇨🇳** 本文的主要贡献总结如下。

*🇬🇧* The main contributions of this work are summarised as follows.

---

**🇨🇳** 1. 我们提出一个面向盲 $T_{60}$ 估计的多任务框架：以 $T_{60}$ 回归为主任务、去噪重建为辅助任务，共享 dual-path time–frequency Conformer backbone（稠密卷积编码器与两个双阶段 Conformer 块），并在共享瓶颈之上分出 Fourier-KAN 回归头与 CMGAN 风格去噪解码分支。据我们所知，这是首次将去噪辅助监督系统性地引入 dual-path time–frequency Conformer 的盲 $T_{60}$ 回归任务。
2. 我们通过梯度探针与表征分析（CKA）揭示了去噪辅助监督的作用机制：两任务梯度方向在整个训练过程中近似正交（$\cos\approx 0$，无系统性对抗冲突），故可联合优化而互不干扰；表征分析表明，去噪任务会重塑共享特征，而这种重塑——而非任何梯度冲突——正是 $T_{60}$ 估计改善的根源。
3. 我们设计了一个轻量化的 Fourier 基 Kolmogorov–Arnold 回归头，以两层 Fourier-KAN（各层使用不同数量的三角基）将多统计量池化特征映射到 $T_{60}$。实验证实，在相同多任务设置下该头优于常规 MLP 回归头。
4. 我们在覆盖仿真/真实 RIR 与已见/未见噪声类型所有组合的四个含噪混响测试集上作系统评估。两任务共享 encoder 与 Conformer 块，$T_{60}$ 估计的 RMSE 相比单任务消融基线（111.15 ms）降低 16.4%（92.91 ms 对 111.15 ms），代价是新增两个去噪解码器（约 0.54M 参数；总参数 1.41M，单任务为 861,793）。所提模型取得平均 RMSE 92.91 ms、MAE 56.68 ms、Pearson 相关系数 0.9625，并优于 BERP、DAMTL、noiseaware。

*🇬🇧* \begin{enumerate}[(1)] We propose a multi-task framework for blind $T_{60}$ estimation that takes $T_{60}$ regression as the main task and denoising reconstruction as an auxiliary task, sharing a dual-path time--frequency Conformer backbone (a dense convolutional encoder and two two-stage Conformer blocks) between the two tasks, with a Fourier-KAN regression head and a CMGAN-style denoising decoder branch tapped off the shared bottleneck. To the best of our knowledge, this is the first work to systematically introduce denoising auxiliary supervision into blind $T_{60}$ regression with a dual-path time--frequency Conformer. Through gradient-probe and representation analysis (CKA), we elucidate the mechanism by which the denoising auxiliary supervision takes effect: the two task gradients remain nearly orthogonal throughout training ($\cos\approx 0$, with no systematic antagonistic conflict), so the two tasks can be optimised jointly without interference; representation analysis shows that the denoising task reshapes the shared features, and this reshaping---rather than any gradient conflict---is the source of the $T_{60}$ improvement. We design a lightweight Fourier-basis Kolmogorov--Arnold regression head that maps the multi-statistic pooled bottleneck features to $T_{60}$ through a two-layer Fourier-KAN block, each layer using a different number of trigonometric bases. Experiments confirm that the proposed head outperforms a conventional MLP head of comparable size under the same multi-task setting. We carry out a systematic evaluation on four noisy reverberant test sets covering all combinations of simulated/measured RIRs and seen/unseen noise types. Sharing the encoder and Conformer blocks between the two tasks, the $T_{60}$ RMSE is reduced by $16.4$\,\% relative to a single-task ablation baseline ($92.91$\,ms vs.\ $111.15$\,ms), at the cost of two denoising decoders ($\approx 0.54$\,M parameters; $1.41$\,M parameters in total vs.\ $861\,793$ for the single-task estimator). The proposed model attains an average RMSE of $92.91$\,ms, a MAE of $56.68$\,ms and a Pearson correlation of $0.9625$, and outperforms BERP, DAMTL and noiseaware. \end{enumerate}

---

**🇨🇳** 本文余下部分组织如下。第 2 节形式化声学信号模型以及 RIR 与 $T_{60}$ 的关系。第 3 节详述所提网络，包括 Conformer backbone、去噪解码分支、多统计量池化层、Fourier-KAN 回归头以及多任务训练目标。第 4 节介绍数据集、基线与实现细节。第 5 节汇报 $T_{60}$ 估计与去噪的定量结果、消融实验，以及梯度与表征分析。第 6 节总结全文。

*🇬🇧* The remainder of this paper is organised as follows. Section~[§] formalises the acoustic signal model and the relation between the room impulse response and $T_{60}$. Section~[§] describes the proposed network in detail, including the Conformer backbone, the denoising decoder branch, the multi-statistic pooling layer, the Fourier-KAN head and the multi-task training objective. Section~[§] presents the datasets, baselines and implementation details. Section~[§] reports the quantitative results for $T_{60}$ estimation and denoising, the ablation studies, and the gradient and representation analyses, and Section~[§] concludes the paper.

---

**🇨🇳** ---

---


# ═══ 信号模型 / Signal model ═══


### 🇨🇳 子节: 2.1 声学信号模型

*🇬🇧* \label{sec:bg}

---

**🇨🇳** 设 $s(t)$ 为静止声源发出的无混响语音，$h(t)$ 为从声源位置到传声器的房间脉冲响应（RIR），$n(t)$ 为传声器处捕获的互不相关加性噪声。在时域上，录制的含噪混响信号 $y(t)$ 建模为


### 🇬🇧 Subsection: Acoustic signal model

---

**🇨🇳** 其中 $\ast$ 表示卷积。对式 (1) 作短时傅里叶变换得到 $Y(k,l) \approx S(k,l)\,H(k,l) + N(k,l)$，其中 $k$、$l$ 分别索引频率窗与时间帧。盲 $T_{60}$ 估计旨在学习一个映射 $\mathcal{F}:Y \mapsto \hat{T}_{60}$，仅以录制信号的复 STFT 作为输入，无法访问 $S$、$H$ 或 $N$。


### 🇬🇧 Subsection: Reverberation time and the Polack model

---


### 🇨🇳 子节: 2.2 混响时间与 Polack 模型

---

**🇨🇳** 要从观测信号 $y(t)$ 估计 $T_{60}$，需要理解 $T_{60}$ 与 RIR $h(t)$ 结构之间的关系。在 Polack 模型下（Polack，1988），晚期 RIR 被描述为零均值阻尼高斯过程，

---

**🇨🇳** 其中 $b(t)$ 是方差为 $\sigma_b^{2}$ 的平稳独立同分布序列，$\delta>0$ 为房间的阻尼常数。由式 (2) 的二阶统计可得平方能量包络的衰减规律

---

**🇨🇳** 混响时间可解析地写为

---

**🇨🇳** 基于模型的估计器通过对（估计得到的）能量包络取对数后作线性回归来拟合衰减速率 $\delta$。然而当混响语音被噪声污染时，包络不再为纯指数，$\delta$ 的估计会产生严重偏置，通常导致 $T_{60}$ 被高估（Gaubitch 等，2012）。这一局限正是使用数据驱动方法、直接学习从 $Y(k,l)$ 到 $\hat{T}_{60}$ 映射的动机。

---

**🇨🇳** ---

---


# ═══ 所提方法 / Proposed method ═══

**🇨🇳** 本节描述所提的多任务 $T_{60}$ 估计网络。如图 1 所示，网络由一个**共享主干**与两个任务分支组成。共享主干包括：（i）一个复数频谱*稠密卷积编码器*，将含噪混响波形转化为高维时频表示；以及（ii）由 $N{=}2$ 个堆叠的*双阶段 Conformer 块*（TSCB；其内部结构见图 2）捕获长程时频依赖。所得瓶颈特征馈入两条并行分支：（iii）*Fourier-KAN 回归头*——将瓶颈输出聚合为单一归一化 $T_{60}$ 估计（主任务）；（iv）*去噪解码分支*（掩码解码器 + 复数解码器）——重建增强语音（辅助任务）。两条分支共享 encoder 与 Conformer 块并联合训练，使去噪梯度作为共享表征的正则化器；训练目标 $\mathcal{L}=\alpha\,\mathcal{L}_{\text{den}}^{*}+\beta\,\mathcal{L}_{T_{60}}^{*}$（任务权重 $(\alpha,\beta)$ 在第 5 节调优）详见 §3.5。

*🇬🇧* \label{sec:method}

---


### 🇨🇳 子节: 3.1 稠密卷积编码器

*🇬🇧* This section describes the proposed multi-task $T_{60}$ estimation network. As illustrated in Fig.~[§], the network is composed of a shared backbone and two task-specific branches. The shared backbone consists of (i) a complex-spectrum \emph{dense convolutional encoder} that turns the noisy reverberant waveform into a high-dimensional time--frequency representation, and (ii) two two-stage Conformer blocks (TSCBs) that capture long-range spectro-temporal dependencies. The resulting bottleneck feeds two parallel branches: (iii) a Fourier-KAN regression head that aggregates the bottleneck output into a single normalised $T_{60}$ estimate (the main task), and (iv) a denoising decoder branch (a mask decoder and a complex decoder) that reconstructs the enhanced speech (the auxiliary task). The two branches share the encoder and the Conformer blocks and are trained jointly, so that the denoising gradient acts as a regulariser on the shared representation; the training objective is detailed in Section~[§].

---

**🇨🇳** 稠密卷积编码器将含噪混响波形转化为两任务共享的高维时频表示。所有录音重采样至 16 kHz、转为单声道，并截断到固定长度 4 s，每条语音 $L=64\,000$ 个样本。为使输入统计量与绝对录音电平无关，每个波形 $y[n]$ 乘以能量归一化因子

*🇬🇧* \begin{figure*}[t] \centering \includegraphics[width=\textwidth]{figs/overall_multitask.png} \caption{Overview of the proposed multi-task Conformer-based $T_{60}$ estimation network. Complex STFT features of the noisy reverberant waveform $x(t)$ are processed by a dense convolutional encoder and $N{=}2$ stacked two-stage Conformer blocks (TSCBs; see Fig.~[§] for the internal block structure), producing a shared bottleneck of shape $B{\times}64{\times}T{\times}F/2$. Three branches diverge from the shared bottleneck. The $T_{60$ Branch} applies channel-wise global statistics pooling ($\mu\,\|\,\max\,\|\,\sigma$) followed by a Linear--LayerNorm projection, a two-layer Fourier-KAN head ($\Omega{=}16,8$; $64{\to}32{\to}16$), a Linear--Sigmoid and an inverse min--max denormalisation to yield $\hat{T}_{60}$. The Mask Decoder and the Complex Decoder reuse the CMGAN dilated DenseNet ($d\in\{1,2,4,8\}$) and sub-pixel upsampling to predict a real-valued magnitude mask and a complex residual $\Delta(real, imag)$. Spectral recombination multiplies the mask with the noisy magnitude (operator $M$), reattaches the noisy phase $\angle X$ and adds the complex residual ($\oplus$) to obtain the reconstructed complex spectrum $\hat{X}$, which is inverted by iSTFT into the enhanced waveform $\hat{s}(t)$. The two tasks are trained jointly with the loss $\mathcal{L}_{total}=\alpha\,\mathcal{L}_{denoise} +\beta\,\mathcal{L}_{T_{60}}$ (the task weights $(\alpha,\beta)$ are tuned in Section~[§]); see Section~[§] for the full definition.} \label{fig:overall} \end{figure*}

---

**🇨🇳** 其中 $\varepsilon=10^{-8}$ 用于数值稳定，训练与推理采用相同归一化。随后以 $N=400$ 个样本的 Hamming 分析窗、$H=100$ 个样本的帧移计算复 STFT，得到 $T=641$ 帧、$F=201$ 个频率窗的复频谱 $\mathbf{Y}\in\mathbb{C}^{T\times F}$；将其实部与虚部堆叠为双通道张量，并拼接幅度 $|\mathbf{Y}|=\sqrt{\mathrm{Re}(\mathbf{Y})^{2}+\mathrm{Im}(\mathbf{Y})^{2}}$ 作为第三通道，得到输入张量


### 🇬🇧 Subsection: Dense convolutional encoder

---

**🇨🇳** 编码器采用三阶段稠密卷积设计。一个逐点卷积首先将三通道输入投影到 $C=64$ 通道，其后接 instance normalization 与 parametric ReLU。一个深度 $D=4$ 的*膨胀稠密块*随后施加四个级联的 $2\times 3$ 卷积，每个卷积的输入由此前所有卷积的输出拼接而成；第 $d$ 个卷积沿时间轴的膨胀率为 $2^{d-1}$，在控制参数量的同时指数级扩大时间感受野。每个卷积后接 instance normalization 与 PReLU 激活。最后，一个步长 $1\times 3$ 卷积将频率轴下采样两倍，得到编码器瓶颈


### 🇬🇧 Subsection: Two-stage Conformer block

---

**🇨🇳** 其中 $C=64$、$T=641$、$F'=101$。


### 🇬🇧 Subsection: $T_{60

---


### 🇨🇳 子节: 3.2 双阶段 Conformer 块

*🇬🇧* The pooled embedding $e\in\mathbb{R}^{192}$ is mapped to the final estimate $\hat{T}_{60}$ by the Fourier-KAN head, depicted in Fig.~[§]. The head is composed of a linear projection, a stack of two Fourier-KAN layers and a final linear output. Each step is described below.

---

**🇨🇳** 编码器瓶颈 $\mathbf{F}^{(0)}$ 随后由 $N=2$ 个堆叠的双阶段 Conformer 块（TSCB）进一步细化。Conformer 块（Gulati 等，2020）将自注意力的全局建模能力与卷积的局部特征提取相结合，已成为语音增强与说话人分离的标准序列建模选择。对长度为 $L$、特征维 $d$ 的序列 $\mathbf{Z}\in\mathbb{R}^{L\times d}$，该块由四个被残差连接包裹的子模块组成：


### 🇬🇧 Subsection: Denoising decoder branch

---

**🇨🇳** 其中 $\mathrm{FFN}_1$、$\mathrm{FFN}_2$ 为两个 pre-normalization 的位置式前馈模块（Swish 激活），$\mathrm{MHSA}$ 为带 Shaw 式相对位置编码（Shaw 等，2018）的多头自注意力模块，$\mathrm{Conv}$ 为带门控线性单元与 batch normalization 的深度可分离一维卷积模块。两个前馈分支的半尺度构成了所谓的 "Macaron" 结构，已被证明能稳定优化（Gulati 等，2020）。


### 🇬🇧 Subsection: Loss function

---

**🇨🇳** 将 Conformer 块应用于语音的时频表示时，通常以 dual-path 方式部署：堆叠两个 Conformer 块，一个沿时间轴、一个沿频率轴运算，共同构成一个两阶段 Conformer 块（TSCB）（Cao 等，2022）。具体地，对输入特征图 $\mathbf{F}\in\mathbb{R}^{B\times C\times T\times F'}$：

*🇬🇧* At inference time the sigmoid output is denormalised by the inverse of Eq.~[式] to yield the final estimate $\hat{T}_{60}\in[T_{\min}, T_{\max}]$.

---

**🇨🇳** 其中 $\mathrm{Reshape}_{t}$ 将频率轴并入 batch 维、把时间轴作为序列轴；Conformer 块随后作用于形状 $(B\,F')\times T\times C$ 的张量。类似地，

---

**🇨🇳** $\mathrm{Reshape}_{f}$ 产生形状 $(B\,T)\times F'\times C$ 的张量。每个 Conformer 块有 $h=4$ 个维 $d_h=C/4=16$ 的注意力头、前馈扩展因子 4、卷积核大小 31 的深度可分离一维卷积，以及注意力与前馈子模块上 0.2 的 dropout。第二个 TSCB 之后，瓶颈特征图记为 $\mathbf{F}^{(N)}\in\mathbb{R}^{C\times T\times F'}$；该瓶颈由 $T_{60}$ 回归头（§3.3）与去噪分支（§3.4）共享。当 $C=64$、$T=641$、$F'=101$ 时，逐块自注意力在 $T$ 或 $F'$ 上为二次复杂度，但得益于 dual-path 分解，在乘积 $T\,F'$ 上仅为线性复杂度——这正是 4 秒窗口能在单 GPU 上可行的原因。

---


### 🇨🇳 子节: 3.3 $T_{60}$ 回归头

---

**🇨🇳** $T_{60}$ 回归头将共享瓶颈 $\mathbf{F}^{(N)}$ 经两步池化与一个 Fourier-KAN 块凝缩为单一的归一化 $T_{60}$ 估计。首先对频率轴求平均，反映 $T_{60}$ 概括的是房间宽带属性而非特定谱型的假设：

---

**🇨🇳** 得到 $\mathbf{Z}\in\mathbb{R}^{C\times T}$。随后由多统计量池化层沿时间维计算均值、最大值与标准差（Snyder 等，2018）：

---

**🇨🇳** 三个统计量携带关于语音能量包络的互补信息：$\mu(\mathbf{Z})$ 概括长期平均电平，$\max(\mathbf{Z})$ 捕获通常与元音起音相关的峰值激励，$\sigma(\mathbf{Z})$ 捕获包络的时间调制，而已知后者与晚期混响尾的衰减速率强相关（Falk 等，2010）。当 $C=64$ 时，所得嵌入有 $3C=192$ 个元素。

---

**🇨🇳** **Kolmogorov–Arnold 基。** Kolmogorov–Arnold 表示定理指出，任意连续多元函数 $f:[0,1]^{n}\to\mathbb{R}$ 都可写作连续一元函数与加法的有限复合。Liu 等（2024）据此提出 Kolmogorov–Arnold 网络（KAN）：用置于网络每条*边*上的可学习一元函数取代多层感知机（MLP）的固定标量非线性。一个输入维 $I$、输出维 $O$ 的 KAN 层实现映射

---

**🇨🇳** 其中每个 $\phi_{j,i}:\mathbb{R}\to\mathbb{R}$ 为可学习一元函数，$b_j$ 为每个输出的偏置。原 KAN 论文以三次 B 样条参数化 $\phi_{j,i}$，表达力强但需要自适应网格更新等簿记工作，在回归头典型的小批量下相对较慢。Fourier-KAN 以截断三角级数取代样条基（作者待核，2024）。对*频率数* $\Omega\in\mathbb{N}$，每条边函数写作

---

**🇨🇳** 于是 Fourier-KAN 层的输出为

---

**🇨🇳** 每层可训练参数数为 $2\,O\,I\,\Omega + O$，是同等 $O\times I$ 线性层的若干倍，但在 $I$、$O$ 较小时绝对量仍然很小。两个性质使该基适合作为本文回归头：（i）权重的梯度是输入的闭式三角函数，无需维护网格；（ii）该基对光滑且近似周期的依赖关系有强归纳偏置，与池化瓶颈特征到 $T_{60}$ 的经验映射形状相符。池化嵌入 $\mathbf{e}\in\mathbb{R}^{192}$ 经线性投影、两层 Fourier-KAN 与一个最终线性输出映射到 $\hat{T}_{60}$，分述如下。

---

**🇨🇳** **投影。** 一个线性投影后接 layer normalization，将池化嵌入降为 $d_{0}=64$ 维向量，

---

**🇨🇳** layer normalization 之后不再施加额外激活：归一化已约束了首个 Fourier 层的输入范围，而 tanh 这类饱和非线性会削弱周期基的有效容量。

---

**🇨🇳** **Fourier-KAN 块。** 投影后的嵌入经 $L=2$ 层 Fourier-KAN 处理，隐层维度 $(d_{1},d_{2})=(32,16)$。首层使用较大的频率数 $\Omega_{0}=16$，使其能直接从嵌入学习细粒度非线性特征；内层使用较小的频率数 $\Omega_{1}=8$ 以提高参数效率。形式上，对 $\ell=1,2$，

---

**🇨🇳** 层输出再经 layer normalization 与概率 0.1 的 dropout 后处理：

---

**🇨🇳** Fourier 权重 $a^{(\ell)}_{j,i,\omega}$、$b^{(\ell)}_{j,i,\omega}$ 以标准差 $\sqrt{1/(d_{\ell-1}\,\Omega_{\ell-1})}$ 的零均值高斯初始化，使层输出方差在初始化时近似为单位量，与输入维和频率数无关。

---

**🇨🇳** **输出。** 一个线性层将 $\mathbf{u}^{(2)}\in\mathbb{R}^{16}$ 映射为标量，经 sigmoid 得到归一化估计

---


### 🇨🇳 子节: 3.4 去噪解码分支

---

**🇨🇳** 为向共享主干注入去噪监督，我们在瓶颈特征 $\mathbf{F}^{(N)}$ 之上分出一条解码分支，采用掩码 + 复数残差结构。一个掩码解码器输出幅度掩码 $\mathbf{M}\in\mathbb{R}^{1\times T\times F}$，与输入幅度相乘得到掩蔽后的幅度 $\hat{M}=|\mathbf{Y}|\odot\mathbf{M}$；一个复数解码器输出复数残差 $\Delta\in\mathbb{R}^{2\times T\times F}$。结合含噪相位 $\angle\mathbf{Y}$，重建增强频谱的实部与虚部为

---

**🇨🇳** 再经 iSTFT 得到增强时域语音。两个解码器均采用与 encoder 对称的稠密设计。

---


### 🇨🇳 子节: 3.5 损失函数

---

**🇨🇳** 为匹配式 (19) 中 sigmoid 的输出范围，每个真值 $T_{60}$ 在送入损失前线性映射到 $[0,1]$：

---

**🇨🇳** $T_{60}$ 回归损失为 $B$ 条语音小批量上预测与真值归一化 $T_{60}$ 之间的均方误差：

---

**🇨🇳** 去噪损失结合压缩域实/虚部损失、幅度损失与时域损失：

---

**🇨🇳** 其中 $\mathcal{L}_{ri}$、$\mathcal{L}_{mag}$ 分别为（功率压缩后）增强频谱实/虚部与幅度相对无噪混响目标 $\tilde{S}$ 的均方误差，$\mathcal{L}_{time}$ 为 iSTFT 增强波形与 $\tilde{s}$ 的 $\ell_{1}$ 损失。

---

**🇨🇳** 由于去噪损失与 $T_{60}$ 损失的原始量级不同，直接加权时权重 $\alpha/\beta$ 难以直观反映两任务对共享层的相对贡献。为使两损失能在同一尺度上加权，我们将二者各自除以自身的指数移动平均（EMA）归一到单位量级，记为 $\mathcal{L}_{\text{den}}^{*}$、$\mathcal{L}_{T_{60}}^{*}$，总损失为

---

**🇨🇳** 其中 $\boldsymbol{\theta}$ 收集共享 encoder 与 TSCB、Fourier-KAN 头及两个去噪解码器的所有可训练参数；去噪损失子权重 $(w_{ri},w_{mag},w_{time})=(0.1,0.9,0.2)$。$(\alpha,\beta)$ 的具体取值及所对应的主模型配置见 §4.2。

---

**🇨🇳** 推理时将 sigmoid 输出按式 (21) 的逆映射反归一化，得到最终估计 $\hat{T}_{60}\in[T_{\min}, T_{\max}]$。

---

**🇨🇳** ---

---


# ═══ 实验 / Experiments ═══


### 🇨🇳 子节: 4.1 数据集

*🇬🇧* \label{sec:exp}

---

**🇨🇳** 实验数据集由 16 kHz 单声道、约 4 秒的含噪混响语音片段组成，$T_{60}$ 覆盖 $[0.1, 1.5]$ s，SNR 覆盖 $-5$ 至 $20$ dB。数据集包含训练集 $40\,000$ 条、验证集 $5\,136$ 条，以及四个各含 $1\,080$ 条的测试集，构成仿真/真实 RIR 与已见/未见噪声的 $2\times 2$ 设计：


### 🇬🇧 Subsection: Dataset

---

**🇨🇳** 其中"已见/未见"指噪声类型是否在训练集中出现。该设计用于评估模型在不同 RIR 真实度与噪声泛化条件下的稳健性。下图为训练池与测试集（按 RIR 类型）的 $T_{60}$ 分布。

*🇬🇧* Experiments are conducted on the T60\_Dataset\_v7\_4s dataset, which consists of $16$\,kHz mono noisy reverberant speech clips of about $4$\,s. Each sample directory speech\{id\\_reverb\_\{T60\}\_\{SNR\}dB/} contains three files: the noisy reverberant recording \{name\.wav} (model input), the noise-free reverberant speech \{name\\_denoised.wav} (denoising target), and the noise component \{name\\_noise.wav} (unused). $T_{60}$ covers $[0.1, 1.5]$\,s and the SNR covers $-5$ to $20$\,dB. The dataset is split into a training set of $40\,000$ clips, a validation set of $5\,136$ clips, and four test sets of $1\,080$ clips each, forming a $2\times 2$ design over simulated/real RIRs and seen/unseen noise (Table~[§]). ``Seen/unseen'' refers to whether the noise type appears in the training set; this design evaluates robustness to RIR realism and to noise generalisation. Figure~[§] shows the $T_{60}$ distribution of the training pool and of the test sets (grouped by RIR type).

---


### 🇨🇳 子节: 4.2 对比方法与评估指标


### 🇬🇧 Subsection: Baselines and metrics

---

**🇨🇳** **对比方法。** 将所提模型与三种盲 $T_{60}$ 估计方法在同一数据集上对比：（i）BERP（Wang 等，2025），采用注意力机制与卷积层相结合的统一房间特征编码器及多个分参数预测器的通用盲房间参数估计框架；（ii）DAMTL（Zhang 等，2024），以去噪为辅助任务的多任务 $T_{60}$ 估计；（iii）noise-aware 时频掩蔽方法（Zheng 等，2022），两阶段框架。三种方法均在 T60\_Dataset\_v7 上以相同划分重新训练/评估，以保证对比公平。

*🇬🇧* **Baselines..**  The proposed model is compared with three blind $T_{60}$ estimators on the same dataset: (i) BERP [wang2025berp], a universal blind room-parameter estimator with an attention-and-convolution unified encoder and separate per-parameter predictors; (ii) DAMTL [zhang2024damtl], a denoising-aided multi-task $T_{60}$ estimator; and (iii) the noise-aware time--frequency masking method [zheng2022noiseaware], a two-stage framework. All three are re-trained and evaluated on T60\_Dataset\_v7 with the same splits for a fair comparison.

---

**🇨🇳** **评估指标。** $T_{60}$ 估计精度与去噪质量采用以下指标评估。

*🇬🇧* **Evaluation metrics..**

---

**🇨🇳** *$T_{60}$ 估计：*

*🇬🇧* $T_{60}$ estimation accuracy and denoising quality are evaluated with the following metrics.

---

**🇨🇳** - **均方根误差（RMSE）**：$\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{T}_{60,i}-T_{60,i})^{2}}$。估计残差的标准差，对大误差惩罚更重，越小越好。$T_{60}$ 以毫秒报告。
- **平均绝对误差（MAE）**：$\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|\hat{T}_{60,i}-T_{60,i}|$。平均绝对偏差，与 $T_{60}$ 同量纲，越小越好。
- **Pearson 相关系数（$r$）**：估计与真值的线性相关，取值 $[-1,1]$，越接近 1 越好。
- **决定系数（$R^{2}$）**：$R^{2}=1-\frac{\sum_{i}(\hat{T}_{60,i}-T_{60,i})^{2}}{\sum_{i}(T_{60,i}-\bar{T}_{60})^{2}}$。估计解释的真值方差占比，取值 $(-\infty,1]$，越接近 1 越好。


### 🇬🇧 Subsection: Model and training setup

---

**🇨🇳** 除整体指标外，还按 $T_{60}$ 区间（$0.1$ s 步长）与 SNR 分 bin 报告 RMSE 与 $r$。

*🇬🇧* The model architecture is detailed in Section~[§] (shared DenseEncoder + two TSCBs + Fourier-KAN regression head + CMGAN denoising decoders); here we only describe the training configuration. All recordings are resampled to $16$\,kHz mono and chunked/padded to $4$\,s. We use the AdamW optimiser (learning rate $5\times10^{-4}$, weight decay $10^{-5}$), batch size $16$, gradient clipping $5.0$, and mixed-precision training for at most $100$ epochs, with early stopping when the validation RMSE does not improve for $15$ epochs. Following Section~[§], the denoising and $T_{60}$ losses are each divided by their own EMA to a common scale and combined as $\mathcal{L}=\alpha\,\mathcal{L}_{den}^{*}+ \beta\,\mathcal{L}_{T_{60}}^{*}$; the task weights $(\alpha,\beta)$ are tuned in Section~[§].

---

**🇨🇳** *去噪：*

*🇬🇧* To check whether the two tasks interfere on the shared layers, every $20$ training steps and before the main backward pass we measure the gradients of the two (normalised but unweighted) losses with respect to the three shared modules (DenseEncoder, TSCB$_{1}$, TSCB$_{2}$) and record the cosine similarity $\cos$ between them.

---

**🇨🇳** - **PESQ**：宽带感知语音质量（ITU-T P.862.2），取值 $-0.5\sim 4.5$，越大越好。
- **STOI**：短时客观可懂度，取值 $0\sim 1$ 的可懂度度量，越大越好。
- **SI-SDR（dB）**：$\mathrm{SI\text{-}SDR}=10\log_{10}\frac{\lVert\mathbf{s}_{\mathrm{target}}\rVert^{2}}{\lVert\mathbf{e}_{\mathrm{noise}}\rVert^{2}}$。尺度不变的重建保真度，越大越好。

---

**🇨🇳** 所有去噪指标均在含噪输入与增强输出上计算并报告提升量。如 §3.4 所述，去噪 target 为无噪混响语音，故这些指标衡量的是去噪（而非去混响）质量。

---


### 🇨🇳 子节: 4.3 模型与训练设置

---

**🇨🇳** 模型结构详见 §3（共享 DenseEncoder + 两 TSCB + Fourier-KAN 回归头 + CMGAN 去噪解码器），此处仅交代训练配置。所有录音重采样至 16 kHz 单声道、截取/补零到 4 秒。采用 AdamW 优化器（学习率 $5\times10^{-4}$、权重衰减 $10^{-5}$），批大小 16，梯度裁剪 $5.0$，混合精度训练，最多 100 个 epoch，验证集 RMSE 连续 15 个 epoch 不下降则早停。按 §3.5 将去噪与 $T_{60}$ 损失各自 EMA 归一化后，以 $\mathcal{L}=\alpha\,\mathcal{L}_{\text{den}}^{*}+\beta\,\mathcal{L}_{T_{60}}^{*}$ 联合训练；任务权重 $(\alpha,\beta)$ 在 §5.1 调优。

---

**🇨🇳** 为检验两任务在共享层上是否相互干扰，每 $20$ 个训练步在主反向传播前测量去噪损失与 $T_{60}$ 损失对三个共享模块（DenseEncoder、TSCB\_1、TSCB\_2）的*归一化但未加权*梯度的余弦相似度 $\cos$。

---

**🇨🇳** ---

---


# ═══ 结果与讨论 / Results and discussion ═══


### 🇨🇳 子节: 5.1 消融实验

*🇬🇧* \label{sec:results}

---

**🇨🇳** **任务权重调优。** 我们首先调优归一化损失 $\mathcal{L}=\alpha\,\mathcal{L}_{\text{den}}^{*}+\beta\,\mathcal{L}_{T_{60}}^{*}$ 的任务权重 $(\alpha,\beta)$。表 2 报告了覆盖去噪主导、持平、$T_{60}$ 主导三档的五组配置的 $T_{60}$ 精度。精度随去噪权重相对 $T_{60}$ 权重的增大而提升，在 $(\alpha,\beta)=(1.0,0.05)$ 取得最优（RMSE 92.91 ms）——故本文以此配置作为主模型。一旦去噪权重降至持平或 $T_{60}$ 主导档，RMSE 退化至约 110 ms，已不优于单任务基线（111.15 ms），证实只有当去噪任务获得明显更大的权重时精度才改善。


### 🇬🇧 Subsection: Ablation study

---

**🇨🇳** **表 2　任务权重消融（归一化损失；test1–test4 平均）。$T_{60}$ 精度随去噪权重比 $\alpha/\beta$ 增大而提升，$(1.0,0.05)$ 最优。**

*🇬🇧* **Task-weight tuning..**  We first tune the task weights $(\alpha,\beta)$ of the normalised loss $\mathcal{L}=\alpha\,\mathcal{L}_{den}^{*}+ \beta\,\mathcal{L}_{T_{60}}^{*}$. Table~[§] reports the $T_{60}$ accuracy for five configurations spanning the denoise-dominant, balanced and $T_{60}$-dominant regimes. Accuracy improves as the denoising weight is increased relative to the $T_{60}$ weight, and the best result is obtained at $(\alpha,\beta)=(1.0,0.05)$ with an RMSE of $92.91$\,ms---we therefore adopt this configuration as the main model throughout the paper. Once the denoising weight is reduced to the balanced or $T_{60}$-dominant regime the RMSE deteriorates to $\sim$$110$\,ms, no better than the single-task baseline ($111.15$\,ms), confirming that accuracy improves only when the denoising task is given a clearly larger weight.

---

**🇨🇳** 为检验两任务是否会相互干扰，我们用 §4.3 的梯度探针测量训练全程中两任务梯度在共享 encoder 上的余弦 $\cos$。如图所示，$\cos$ 始终集中在 0 附近（$|\cos|$ 均值约 0.10，仅约 41% 的步数 $\cos<0$）：两任务梯度近似正交，既非协同对齐也非系统性冲突。因此两任务可联合优化而互不干扰，决定 $T_{60}$ 精度的是任务权重比——即允许去噪任务在多大程度上塑造共享表征——而非任何梯度冲突。这种塑造在表征层面如何发生，见 §5.2。

*🇬🇧* **Regression-head ablation..**  We further compare the Fourier-KAN regression head against an MLP head of comparable capacity, both under the same multi-task configuration $(\alpha,\beta)=(1.0,0.05)$.

---

**🇨🇳** **回归头消融。** 我们进一步在相同的多任务配置 $(\alpha,\beta)=(1.0,0.05)$ 下，对比 Fourier-KAN 回归头与相近容量的 MLP 头。

*🇬🇧* To check that the two tasks do not interfere antagonistically, we use the gradient probe of Section~[§] to measure the cosine $\cos$ between the two task gradients on the shared encoder throughout training. As Fig.~[§] shows, $\cos$ stays centred near $0$ (mean $\lvert\cos\rvert\approx 0.10$, with only $\sim$$41$\,\% of steps having $\cos<0$): the two task gradients are nearly orthogonal, neither cooperatively aligned nor in systematic conflict. The two tasks can therefore be optimised jointly without one interfering with the other, and what governs $T_{60}$ accuracy is the task-weight ratio---i.e.\ how much the denoising task is allowed to shape the shared representation---rather than any gradient conflict. How this shaping actually takes place at the representation level is examined in Section~[§].

---

**🇨🇳** **表 3　回归头消融（多任务设置 $(\alpha,\beta)=(1.0,0.05)$）**


### 🇬🇧 Subsection: Representation analysis

---


### 🇨🇳 子节: 5.2 表征分析

*🇬🇧* To examine how the denoising auxiliary supervision reshapes the shared representation, we apply CKA (feature similarity between each multi-task configuration and the single-task baseline, per layer) and linear probing (freeze the model, read out $T_{60}$ by linear regression per layer, and report the linearly-separable $T_{60}$ information as $R^{2}$). CKA is computed deterministically over the full feature Gram matrix and is therefore stable; the linear-probe $R^{2}$ uses a random train/test split and is reported as a qualitative cross-check rather than a point comparison.

---

**🇨🇳** 为说明去噪辅助监督如何重塑共享表征、进而惠及 $T_{60}$，我们对各共享层做 CKA（比较各多任务配置与单任务基线在各层的特征相似度）与线性探针（冻结模型，以线性回归读出 $T_{60}$，看各层表征的 $T_{60}$ 可分性 R²）。CKA 基于完整特征 Gram 矩阵作确定性计算，数值稳定；线性探针 R² 采用随机 train/test 划分，存在一定抖动，故仅作定性交叉验证，不作逐点比较。

*🇬🇧* CKA reveals a clear, depth-graded pattern. At the encoder all three multi-task regimes remain nearly identical to the single-task baseline (CKA $0.988$--$0.995$): the low-level magnitude/phase encoder is largely task-agnostic. Moving deeper, TSCB$_1$ and TSCB$_2$ diverge, and the divergence increases with the strength of the denoising gradient: at TSCB$_2$, the denoise-dominant regime ($\alpha{=}1.0,\beta{=}0.1$) departs furthest from the single-task solution (CKA $0.699$), the balanced regime is intermediate ($0.743$), and the $T_{60}$-dominant regime ($\alpha{=}0.1,\beta{=}1.0$), in which the denoising gradient is suppressed, stays closest ($0.821$). This ordering of CKA matches the ordering of $T_{60}$ accuracy in Table~[§] (denoise-dominant best, $T_{60}$-dominant worst): the more the denoising gradient reshapes the shared representation, the lower the $T_{60}$ RMSE. In other words, \emph{a stronger denoising gradient does more than add a second objective---it actively reshapes the shared representation, and this reshaping is what the $T_{60}$ head benefits from.}

---

**🇨🇳** CKA 揭示了一个清晰的、随深度递进的规律。在 encoder 层，三档多任务配置都与单任务基线几乎一致（CKA 0.988–0.995）：底层幅度/相位编码器很大程度上与任务无关。向深层推进，TSCB-1 与 TSCB-2 显著偏离，且偏离幅度随去噪梯度强度**递增**：在 TSCB-2 上，去噪主导档（$\alpha{=}1.0,\beta{=}0.1$）偏离最远（CKA 0.699），持平档居中（0.743），而抑制了去噪梯度的 $T_{60}$ 主导档（$\alpha{=}0.1,\beta{=}1.0$）最贴近单任务解（0.821）。这一 CKA 排序与表 2 的 $T_{60}$ 精度排序完全一致（去噪主导最优、$T_{60}$ 主导最差）：去噪梯度对共享表征的重塑越强，$T_{60}$ RMSE 越低。换言之，*更强的去噪梯度所做的不仅是叠加第二个目标——它主动重塑了共享表征，而正是这种重塑让 $T_{60}$ 头受益。*

*🇬🇧* Linear probing (Fig.~[§]) confirms that this restructuring does not discard the $T_{60}$ information: across all four models the deeper layers retain comparable $T_{60}$ decodability (TSCB$_2$ $R^{2}$ in the $0.55$--$0.70$ range), so the denoising supervision does not simply erase the $T_{60}$ cue. The contrast with CKA is informative: the multi-task representations are measurably different from the single-task one (low CKA at TSCB$_2$) yet still carry the $T_{60}$ signal at a comparable level---i.e.\ denoising re-encodes, rather than removes, the $T_{60}$ cue. The actual $T_{60}$ head reads these features through a max/std multi-statistic pool followed by a Fourier-KAN non-linear layer, which can access the re-structured representation that the denoise-dominant regime has produced; this is consistent with the regression-head ablation of Section~[§], where the Fourier-KAN head outperforms an MLP head of comparable capacity under the same multi-task setting.

---

**🇨🇳** 线性探针（图 7）进一步确认，这种重塑并未丢弃 $T_{60}$ 信息：四个模型在深层都保持了相当的 $T_{60}$ 可解码性（TSCB-2 的 R² 都落在 0.55–0.70 区间），说明去噪监督并未简单抹除 $T_{60}$ 线索。线性探针与 CKA 的对照颇有启发：多任务表征与单任务表征可度量地*不同*（TSCB-2 低 CKA），却仍承载着相当水平的 $T_{60}$ 信号——即去噪是把 $T_{60}$ 线索**重新编码**而非**移除**。真实的 $T_{60}$ 头通过 max/std 多统计量池化 + Fourier-KAN 非线性层读取这些特征，能够访问去噪主导档所重塑出的新表征；这与 §5.1 的回归头消融一致——在该消融中 Fourier-KAN 头优于相近容量的 MLP 头。


### 🇬🇧 Subsection: Results comparison with baselines

---


### 🇨🇳 子节: 5.3 与基线的结果对比

*🇬🇧* We compare the proposed model (normalised main model, denoise-dominant $\alpha=1.0,\beta=0.05$, determined by the weight tuning of Section~[§]) with three blind $T_{60}$ estimators (BERP, noiseaware, DAMTL) and the single-task ablation baseline on the four test sets; the averaged metrics are reported in Table~[§].

---

**🇨🇳** 我们将所提模型（归一化主模型，去噪主导档 $\alpha=1.0,\beta=0.05$，由 §5.1 权重调优确定）与三种盲 $T_{60}$ 估计方法（BERP、noiseaware、DAMTL）及单任务消融基线，在同一数据集的四个测试集上对比，平均指标见表 4。

*🇬🇧* The proposed model outperforms the three comparison methods and the single-task baseline on all four test sets (test1--4 = simulated/real RIR $\times$ seen/unseen noise), confirming the effectiveness and cross-condition robustness of the denoising auxiliary task with normalised multi-task training.

---

**🇨🇳** **表 4　主对比结果（T60\_Dataset\_v7，test1–test4 平均；RMSE/MAE 单位 ms）**


### 🇬🇧 Subsection: Effect of SNR on estimation accuracy

---

**🇨🇳** 所提模型在四个测试集（test1–4 = 仿真/真实 RIR × 已见/未见噪声）上均优于三种对比方法与单任务基线，RMSE 相比单任务基线降低 16.4%（92.91 ms vs 111.15 ms），验证了"去噪辅助任务 + 归一化多任务训练"的有效性与跨场景稳健性。

*🇬🇧* Table~[§] reports the per-SNR-bin RMSE of the proposed model and the three baselines, broken down by test-set scenario (simulated vs.\ real RIR).

---


### 🇨🇳 子节: 5.4 信噪比对估计精度的影响


### 🇬🇧 Subsection: Effect of reverberation time on estimation accuracy

---

**🇨🇳** 表 5 报告所提模型与三种基线方法在各 SNR 区间的 RMSE（test1–test4 平均）。

*🇬🇧* Table~[§] reports the per-$T_{60}$-bin RMSE of the proposed model and the three baselines, broken down by test-set scenario.

---

**🇨🇳** **表 5　各 SNR 区间的 RMSE (ms)。每个场景合并同 RIR 类型的两个测试集；"Avg."为四测试集平均。每格最优加粗。**


### 🇬🇧 Subsection: Denoising results

---


### 🇨🇳 子节: 5.5 混响时间对估计精度的影响

*🇬🇧* Table~[§] reports the denoising quality of the three normalised regimes, averaged over test1--test4. Per the target defined in Section~[§], these metrics reflect noise removal while preserving the reverberation.

---

**🇨🇳** 表 6 报告所提模型与三种基线方法在各 $T_{60}$ 区间的 RMSE，按测试集场景分组。

*🇬🇧* The proposed denoise-dominant model removes additive noise while preserving the reverberation, yielding gains of $+0.71$ PESQ, $+0.08$ STOI and $+5.07$\,dB SI-SDR over the noisy input. Crucially, the denoising quality follows the same ordering as the $T_{60}$ accuracy (Table~[§]): the denoise-dominant regime is best at both tasks, and the $T_{60}$-dominant regime is worst at both. There is therefore no trade-off between the auxiliary and the main task---giving the denoising task a larger weight improves the two tasks jointly, consistent with the task-weight ablation of Section~[§].

---

**🇨🇳** **表 6　各 $T_{60}$ 区间 (s) 的 RMSE (ms)。每个场景合并同 RIR 类型的两个测试集；"Avg."为四测试集平均。每格最优加粗。**

---


### 🇨🇳 子节: 5.6 去噪结果

---

**🇨🇳** 表 7 报告三档归一化配置的去噪质量（test1–test4 平均）。按 §3.4 定义的目标，这些指标反映在保留混响前提下的噪声去除。

---

**🇨🇳** **表 7　三档归一化配置的去噪指标（test1–test4 平均）**

---

**🇨🇳** 所提去噪主导模型在保留混响的前提下去除加性噪声，相比含噪输入获得 +0.71 PESQ、+0.08 STOI、+5.07 dB SI-SDR 的提升。关键的是，去噪质量的排序与 $T_{60}$ 精度的排序（表 2）**完全一致**：去噪主导档两项任务都最优，$T_{60}$ 主导档两项任务都最差。因此辅助任务与主任务之间不存在权衡——给去噪任务更大权重能同时改善两任务，这与 §5.1 的任务权重消融结果一致。

---

**🇨🇳** ---

---


# ═══ 结论 / Conclusion ═══

**🇨🇳** 本文提出一个面向盲 $T_{60}$ 估计的多任务学习框架，以 $T_{60}$ 回归为主任务、去噪重建为辅助任务，共享 dual-path time–frequency Conformer backbone，并以 Fourier-KAN 回归头作 $T_{60}$ 估计。主要结论如下：（1）去噪辅助任务以约 0.54M 的解码器开销，使 $T_{60}$ 估计 RMSE 相比单任务消融基线（111.15 ms）降低 16.4%，并优于 BERP、DAMTL、noiseaware 等方法；（2）梯度探针表明，两任务梯度方向近似正交、无系统性冲突，故可联合优化而互不干扰，且任务权重消融显示去噪任务权重越大 $T_{60}$ 精度越高；（3）Fourier-KAN 回归头在相近设置下优于 MLP 头。表征分析（CKA 与线性探针）进一步表明，去噪任务主动**重塑**了共享表征：其权重越大，共享层偏离单任务解越远（CKA 越低），而 $T_{60}$ 线索仍保持相当的线性可解码性——即去噪是把 $T_{60}$ 信息重新编码而非移除，这种重塑后的表征正是 Fourier-KAN 头能利用、从而获得最低 RMSE 的配置。

*🇬🇧* \label{sec:conclusion}

---

*🇬🇧* We proposed a multi-task learning framework for blind $T_{60}$ estimation that pairs $T_{60}$ regression (main task) with denoising reconstruction (auxiliary task) over a shared dual-path time--frequency Conformer backbone, with a Fourier-KAN head for $T_{60}$ estimation. The main findings are: (1) the denoising auxiliary task, at the cost of about $0.54$\,M decoder parameters, reduces the $T_{60}$ RMSE substantially relative to the single-task ablation baseline ($111.15$\,ms) and outperforms BERP, DAMTL and noiseaware; (2) gradient probing shows that the two task gradients are nearly orthogonal with no systematic conflict, so the two tasks can be optimised jointly without interference, and a task-weight ablation shows that giving the denoising task a larger weight improves $T_{60}$ accuracy; (3) the Fourier-KAN head outperforms an MLP head under comparable settings. Representation analysis (CKA and linear probing) further shows that the denoising task actively reshapes the shared representation: the larger its weight, the further the shared layers depart from the single-task solution (lower CKA), yet the $T_{60}$ cue is preserved at a comparable level of linear decodability---i.e.\ denoising re-encodes rather than removes the $T_{60}$ information, and this reshaped representation is what the Fourier-KAN head exploits, yielding the lowest RMSE.

---

*🇬🇧* Future work may explore finer-grained weight sweeps and adaptive gradient balancing (e.g., GradNorm), replacing or extending denoising with other auxiliary tasks (e.g., dereverberation, sound-event detection), and deployment evaluation on real-world recordings.

---

*🇬🇧* \bibliographystyle{cas-model2-names} \bibliography{refs}

---

