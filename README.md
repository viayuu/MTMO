# MTMO

MVMoE 与 PMOCO 的研究初始化工作区。两个项目的作者源码按下列提交原样保存；新增的下载工具、来源清单与说明位于仓库根目录和 `downloads/`。

| 项目 | 上游来源 | 固定提交 |
| --- | --- | --- |
| [Routing-MVMoE](Routing-MVMoE/) | [RoyalSkye/Routing-MVMoE](https://github.com/RoyalSkye/Routing-MVMoE) | `af29e5af0595f94f3ecc3bc46d72df1089a62682` |
| [PMOCO](PMOCO/) | [Xi-L/PMOCO](https://github.com/Xi-L/PMOCO) | `8e734c819b0adaa2851c51ac800df5c5c7d5bbb6` |

## 保存范围

Git 跟踪两个项目的源码、原始 README 和 MIT 许可证、小型方法示意图、论文引用、下载脚本和校验清单。模型、数据集、基线结果、论文 PDF、论文源码压缩包及解压目录、海报和运行缓存由 [.gitignore](.gitignore) 排除，并保留在下载机器本地。

作者的原始 Git 历史在下载机器的 `.upstream-git/` 中保存。MTMO 中的源码为普通文件，便于后续开发；上游来源与提交见 [UPSTREAMS.json](UPSTREAMS.json)。[.gitattributes](.gitattributes) 禁用换行转换，保留作者文件的原始字节。

## 资源与恢复

- [下载记录和本地资源位置](DOWNLOADS.md)
- [逐文件来源与 SHA-256](downloads/file_manifest.csv)
- [论文引用](papers/references.bib)

获取被忽略的上游资源时，在仓库根目录执行：

```bash
conda activate easynco
bash downloads/download_repositories.sh
python downloads/download_papers.py | tee downloads/logs/papers.log
```

第一个脚本从固定的上游提交恢复本地缺失文件，已有文件保持原状；第二个脚本从官方出版平台下载论文和 LaTeX 源码包。两者的日志保存在 `downloads/logs/`。

MVMoE 官方仓库包含 48 个合成数据集、83 个基线结果文件、CVRPLIB/Solomon 实例和 9 个预训练模型。PMOCO 包含 12 个预训练模型；训练和测试实例由作者的 `get_random_problems` 在线生成，仓库没有独立静态数据包。

原项目的使用和引用方式见各自的 README；原始 MIT 许可证分别保存在 [Routing-MVMoE/LICENSE](Routing-MVMoE/LICENSE) 与 [PMOCO/LICENSE](PMOCO/LICENSE)。
