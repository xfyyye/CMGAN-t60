# DACT-KAN 论文审阅快照 · 2026-10-27

## 当前入口

- 中文最终稿：`docs/main_zh.md`。
- 中英逐段对照：`docs/main_bilingual.md`；英文投稿源码：`docs/main.tex`。
- 中文 Word：`docs/main_zh.docx`；书目数据库：`docs/refs.bib`。
- 当前稿件图1–8使用9个图片文件（图6含两张），源图与 `docs/figs_chapter/` 副本逐一相同。4个模块图的最终 SVG 与对应绘图脚本一并保存；本次未重画图。

本次仅将中文稿中的 Xiong 等（2018）及双语稿中对应中英文年份改为2019，与已核实的正式卷期年份、Word和refs.bib一致。在线发表年份为2018，BibTeX键 `xiong2018ercnn` 不重命名，main.tex引用命令不变。`before_after.md` 和 `xiong_year_sync.diff` 为本次文字改动。双语稿顶部的 source sha256 是历史初始翻译冻结版本标识；当前版本的文件哈希以本目录 manifest.json 为准。

## Git 范围与记录

本次保存截至当前已确认的中英文稿件、Word、书目、正式图件、Overleaf副本及必要文字记录。未纳入RIR代码/训练产物、旧图、渲染缓存、重复备份及历史修订中间稿。`docs/HANDOVER.md` 仅暂存本次新增的论文状态段；本地原有RIR交接修改继续保留、未纳入本次提交。

`manifest.json` 为当前稿件/图件哈希与依赖；`checks.json` 为本次核验结果；`snapshot_from_previous_commit.diff` 为相对操作前HEAD的论文文本差异（Word和PNG使用manifest追踪）。`prior_*` 文件是本次快照中此前已完成的Word、书目与TeX格式改动记录，保留其历史检查语境，不表示重新执行了当时的全部检查。

## Overleaf

将 `docs/main.tex`、`docs/refs.bib` 及完整 `docs/figs_chapter/` 上传到同一Overleaf项目根目录（保留figs_chapter层级）。保留已有CAS模板支持文件。当前文类 `cas-sc`、文献样式 `elsarticle-num-names`；仓库未附本地cls/bst，不能将这三个项目当作独立完整模板包。

核验脚本是本次工作区的操作记录，依赖留在本地的操作前备份和暂存清单，不作为独立发布工具。

本轮检查年份、改动范围、图件/引用依赖、标签和对象数量及Git差异空白；未重新执行整篇翻译流程、科学评估、Word渲染或LaTeX编译。此前Word已完成23页视觉检查，此次Word字节不变。

已做 LaTeX 静态检查，未编译；最终排版及运行时兼容性由作者在 Overleaf 验证。作者完整地址、经费、作者贡献及其他声明等待作者/老师提供事实；本快照供老师审阅。

独立只读核查：42个暂存文件与精确清单一致；HANDOVER仅新增论文状态8行，未纳入本地RIR段落；全部暂存科学文件与工作区一致，资源与引用闭合。Git空白检查明确排除`.diff`记录文件，其补丁上下文空行以一个空格为前缀，保持原始patch内容。
