# MVMoE 与 PMOCO 下载清单

下载日期：2026-11-17（北京时间）。本地工作区：`/public/home/shiys/MTMO`。

| 项目 | 代码 | 数据与结果 | 预训练模型 |
| --- | --- | --- | --- |
| [Routing-MVMoE](Routing-MVMoE) | 完整 Git 仓库，含训练、测试、数据生成与基线脚本 | 16 类 VRP × 50/100/200 节点，共 48 个合成数据文件、48,000 个实例；83 个基线结果文件；28 个 CVRPLIB 实例；56 个 Solomon 实例 | 9 个 `.pt` 文件 |
| [PMOCO](PMOCO) | 完整 Git 仓库，含 MOTSP、三目标 MOTSP、MOCVRP、MOKP | 作者代码在训练和测试时调用 `get_random_problems` 在线随机生成实例；当前仓库及 Releases 未提供独立静态数据包 | 12 个 `.pt` 文件 |

## 论文与附件

| 项目 | 本地文件 | 官方来源 |
| --- | --- | --- |
| MVMoE，ICML 2024 正式版，21 页（含附录） | [PDF](papers/MVMoE/MVMoE_ICML2024.pdf) | [PMLR](https://proceedings.mlr.press/v235/zhou24c.html) |
| MVMoE，arXiv 2405.01029v2，21 页 | [PDF](papers/MVMoE/MVMoE_arxiv_2405.01029v2.pdf)；[LaTeX 源码包](papers/MVMoE/MVMoE_arxiv_2405.01029v2_source.tar.gz)；[解压后源码](papers/MVMoE/MVMoE_latex_source) | [arXiv v2](https://arxiv.org/abs/2405.01029v2) |
| PMOCO，ICLR 2022 最终定稿，arXiv 2203.15386v2，31 页（含附录） | [PDF](papers/PMOCO/PMOCO_arxiv_2203.15386v2.pdf)；[LaTeX 源码包](papers/PMOCO/PMOCO_arxiv_2203.15386v2_source.tar.gz)；[解压后源码](papers/PMOCO/PMOCO_latex_source) | [arXiv v2](https://arxiv.org/abs/2203.15386v2)；[OpenReview](https://openreview.net/forum?id=QuObT9BTWo) |

MVMoE 的 [ICML 海报 PDF](Routing-MVMoE/assets/Poster_ICML24.pdf)、海报图片、方法示意图及公开评审记录均随仓库保存。两篇论文的引用保存在 [references.bib](papers/references.bib)。

PMOCO 的 OpenReview PDF 直链返回 HTTP 403，因此保存的是 arXiv 标注的最终定稿版本；该文件首页也标明 “Published as a conference paper at ICLR 2022”。

## 版本与核验

| 仓库 | 来源 | 下载的 main 提交 |
| --- | --- | --- |
| Routing-MVMoE | [RoyalSkye/Routing-MVMoE](https://github.com/RoyalSkye/Routing-MVMoE) | `af29e5af0595f94f3ecc3bc46d72df1089a62682` |
| PMOCO | [Xi-L/PMOCO](https://github.com/Xi-L/PMOCO) | `8e734c819b0adaa2851c51ac800df5c5c7d5bbb6` |

下载核验时，两个仓库工作区干净，`git fsck --full` 通过。初始化 MTMO 后，作者的完整 Git 历史保留在本机 `.upstream-git/`，由根目录 `.gitignore` 排除。21 个预训练模型均为实际文件，无待下载的 Git LFS 指针。论文经过 PDF 文件头、大小、页数和首页标题检查；两个 LaTeX 压缩包均可读取并已解压。48 个合成数据文件逐个读取核验，每个包含 1,000 个实例。

- [机器可读下载清单](downloads/download_manifest.json)：来源、提交、论文版本及下载状态。
- [逐文件清单](downloads/file_manifest.csv)：路径、分类、字节数、SHA-256 与来源 URL。
- [SHA256SUMS](downloads/SHA256SUMS)：在本目录运行 `sha256sum -c downloads/SHA256SUMS` 可复核文件。
- [数据实例统计](downloads/dataset_instances.json)：每个合成数据文件的任务、节点数和实例数。
- [下载日志](downloads/logs)：仓库与论文下载日志。

## 数据和模型位置

- MVMoE 数据：`Routing-MVMoE/data/`；预训练模型：`Routing-MVMoE/pretrained/`。
- PMOCO 模型：各问题目录的 `POMO/result/`，包含 MOTSP 20/50/100、三目标 MOTSP 20/50/100、MOCVRP 20/50/100、MOKP 50/100/200。
- PMOCO 数据生成入口：各问题目录的 `*ProblemDef*.py`；原始测试脚本直接调用相应生成函数。

## Git 提交范围

MTMO 跟踪作者源码、原始 README 与许可证、辅助下载工具、引用及清单。数据集、权重、论文 PDF、LaTeX 压缩包及解压目录、海报和日志保留在本机并由 `.gitignore` 排除。上表中这些资源的本地链接需要先下载对应文件。

源码目录中的作者文件未编辑；根目录说明、`UPSTREAMS.json` 和 `downloads/` 内的工具与记录是本工作区新增内容。可使用 `bash downloads/download_repositories.sh` 从固定提交恢复缺失的原始文件。
