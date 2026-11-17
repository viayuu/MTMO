# MTMO 计算环境

验证日期：2026-11-18。沿用现有 easynco，不重建环境。

- 登录节点：admin01，本地没有 NVIDIA 设备；CUDA 命令须进入计算节点。
- 已有 Slurm allocation：job 1465，gpu03，2 × RTX3090 24GB，10 CPU、20GB RAM。
- 计算环境：Python 3.9，PyTorch 2.5.1+cu124，CUDA 12.4。
- 实际验证：两张3090均完成CUDA kernel；完整E128/六层/N100/globalbatch128双rank前向、反向与更新通过，峰值约12GB/卡。

运行模板：

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/MTMO
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
srun --jobid=1465 --overlap -N1 -n1 -c4 --gres=gpu:2 --immediate=60 python <入口和参数>
```

完整流水线在 tmux 运行 `bash scripts/run_full_mtmo.sh`：正式六任务尺度校准 → 双卡DDP完整训练 → 两卡分片评测16任务×三个规模×101偏好×1000实例×8倍增强 → 路线审计与报告。

训练固定100M实例，使用最终5000epoch checkpoint，不根据未见任务指标选模。N50/200是N100模型的规模泛化评测。正式运行使用冻结的源码快照；再次执行流水线会从已有checkpoint恢复并校验已完成评测的来源/hash。失败会写入status.json并停止，不能把失败或小规模smoke记为完整训练。

进度命令：`python scripts/mtmo_status.py runs/mtmo_n100_seed2023`。
