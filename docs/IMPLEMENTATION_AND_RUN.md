# MTMO 实现与完整运行

2026-11-18。实施基于当前 backbone 设计，副本为 `MTMO/`；原 `Routing-MVMoE/` 与 `PMOCO/` 保留原样。

## 基线与改动

先完成未修改副本提交 `cbd9a1c` 并推送 GitHub，再开始实现。复制核验覆盖264个本地文件；38个可上传作者文件与上游源文件的Git blob及权限一致。模型、数据和论文等大文件继续忽略。

保留原编码器、MoE路由/dispatcher/aux loss、十六任务环境约束与原POMO构造。新增偏好超网络生成Q/K/V及四专家W/b、两个累计目标状态、固定训练任务尺度校准、双目标TCH训练、逐实例偏好解集。环境通过薄包装处理，不维护另一套mask。

两张3090使用DDP，全局batch128、每rank64；每批共享任务和偏好。MoE辅助统计按rank计算并平均，此实现细节记录在run配置中。

## 已执行的验证

- 15项CPU检查通过，含16环境成本/合法性、手工最长路线、open/round/padding、动态专家与梯度、精确断点恢复、真实候选/HV。
- 3090上的模型前向反向、双rank训练检查通过。
- 完整尺寸N100/P100/E128/六层/globalbatch128三步通过，无OOM或非有限梯度；峰值allocated约11.94GiB/卡。
- 实例评测集成已检查CVRP增强1/8、backhaul POMO裁剪及GPU上的OVRPBLTW，保存路线均经原环境重放。
- 小规模检查使用独立临时目录，不计入正式训练和最终结果。

## 完整执行协议

权威配置为 `configs/full_n100.json`：

- 正式尺度校准：六个训练任务各256个在线实例、原MVMoE参考策略、正确路线目标，冻结后所有任务/规模共用。
- 从头训练：N100、原六任务、5000 epochs × 20000实例 = 100,000,000实例，seed2023，全局batch128，原Adam及学习率衰减。
- 最终评测：固定epoch5000模型；16任务 × N50/100/200 × 每任务规模1000实例 × 101偏好 × 8倍增强。
- N100是主评测，N50/200是同一模型的规模泛化，不称为这些规模的训练结果。原固定数据沿用，无额外划分或按未见任务调尺度。
- 每条保存路线均进行客户覆盖、原环境mask重放与成本独立核验。逐实例算前沿/HV，训练参考点与HV参考点分开。
- 完整训练和全部48个任务规模结果核验后才生成最终报告，不用smoke结果填表。

## 命令和文件

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/MTMO
python scripts/supervise_mtmo.py --run_dir runs/mtmo_n100_seed2023
```

长任务放在专用tmux。计算命令统一通过已有Slurm job1465进入gpu03，两卡已核验。完整源码、原参考模型源码和checkpoint在run目录冻结，训练评测都使用该快照。

- `status.json`：流水线阶段。
- `scales.json`：正式尺度和出处。
- `training/train.jsonl`：真实训练统计与吞吐。
- `training/latest.pt`：原子更新的断点；每100epochs另存快照。
- `evaluation/`：逐实例目标、路线、hash和审计结果。
- `report/REPORT.md`：完成后生成的报告、CSV与图。

进度：`python scripts/mtmo_status.py runs/mtmo_n100_seed2023`。

重启采用相同命令，会恢复冻结源码和精确训练断点，并检查已完成评测的配置、源码、数据和结果hash。监督器只对有限次数的调度/网络暂时故障恢复；数值错误、目标错误和OOM会保留错误并停止。

完整训练是数天级任务，不能把“代码检查通过/任务启动”写成“完整训练评测已经完成”。
