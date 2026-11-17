# MTMO backbone 方案：偏好条件稀疏 MoE

日期：2026-11-18。状态：方案设计，尚未实施。

本稿替代 `MTMO_BACKBONE_V0_PLAN.md`。设计标准改为：保证目标、状态、条件计算与训练逻辑合理，并继承两篇论文的核心机制；实现工作量作为工程成本考虑，不决定保留功能缺口。

研究范围仍是搭建 backbone：以 MVMoE 统一多 VRP 任务的表示、MoE 和环境为主体，引入 PMOCO 的偏好条件参数生成与多目标强化学习。本稿不提出额外问题族或声称研究性能已被证明。

固定上游：MVMoE `af29e5af0595f94f3ecc3bc46d72df1089a62682`；PMOCO `8e734c819b0adaa2851c51ac800df5c5c7d5bbb6`。

## 1. 当前默认与继承关系

| 部分 | 当前决定 | 依据 |
| --- | --- | --- |
| 主体模型 | 原版 MVMoE/4E，显式 `model_type=MOE` | 使用原论文已有主模型配置；不依赖 CLI 的 MOE_LIGHT 隐含默认 |
| Encoder | 原输入表示、六层 attention/MoE、InstanceNorm、残差；偏好无关，正常训练 | MVMoE 的共享多任务表示 + PMOCO 的共享 encoder |
| Decoder attention | 超网络生成 Q/K/V | PMOCO 的偏好条件投影 |
| Decoder output | 超网络生成四个线性专家各自的 W、b，保留稀疏 gate/Top2/dispatcher | 将 PMOCO 的单个条件输出投影扩展到 MVMoE 已有专家结构 |
| 动态状态 | 原四属性 + 累计有效总距离 + 迄今最长路线前缀距离 | 使多目标 TCH 的累计进度进入策略 |
| Hypernetwork | 两个 128 维隐藏层与 ReLU；分组 latent 和生成头 | 采用 PMOCO 论文的非线性 MLP，同时复用其参数压缩组织方式 |
| 目标 | 总路线距离、最长单车路线距离，均最小化 | PMOCO MOCVRP 论文定义 |
| 标量化 | 固定正尺度下的 weighted Tchebycheff，零 ideal | 保留 PMOCO 方法，明确目标单位 |
| 偏好分布 | 双目标 uniform simplex：`u~U(0,1); lambda=(u,1-u)` | PMOCO 论文的均匀偏好定义 |
| 训练组织 | 每 batch 一个任务、一个偏好；POMO REINFORCE + 原 MoE 均衡损失 | 两方法兼容的原训练范式 |

上述条件专家、累计状态及尺度约定是本次融合的明确设计，不能写成任一原论文已经原样提供。当前代码与论文不一致时，按功能和定义选择并记录来源。

## 2. 输出投影冲突：直接形成偏好条件 MoE

### 2.1 统一计算图

令节点编码为 H，当前 decoder 上下文为 c_t，偏好为 lambda。共享编码器输出 H；超网络在 rollout 前生成：

\[
W_Q(\lambda),W_K(\lambda),W_V(\lambda),
\{W_{O,e}(\lambda),b_{O,e}(\lambda)\}_{e=1}^{E}.
\]

\[
z_\lambda=\operatorname{MHA}
\bigl(W_Q(\lambda)c_t,W_K(\lambda)H,W_V(\lambda)H;\mathrm{mask}\bigr),
\]

\[
y=\sum_{e\in\operatorname{Top2}(G(z_\lambda))}
g_e(z_\lambda)\,[W_{O,e}(\lambda)z_\lambda+b_{O,e}(\lambda)].
\]

之后沿用 MVMoE 原单头打分、clipping、mask 和 softmax。Q/K/V 公式为概念表达，代码权重统一采用 `F.linear` 的 `[out,in]` 约定。

该结构保留了两个方法在同一位置的功能：偏好控制输出变换，MoE 在多个输出变换之间稀疏选择。每个专家有独立的参数生成头，专家身份由训练学习，不手工绑定到某个目标。

Q/K/V 保留原来的无应用 bias 形式；输出专家保留 MVMoE 的仿射形式，但 W 与 b 都由超网络生成。动态 bias 是保留原仿射专家并对其完整参数条件化的融合选择，不归因于 PMOCO 原代码。保留静态可训练 bias 也能构成合理对照，本稿不声称动态 bias 性能必然更优。

Gate 参数仍是所有偏好共享的普通可训练参数。由于输入 z_lambda 已依赖偏好，gate 已能间接感知偏好；第一版没有额外设计直接拼接 lambda 的 router 的必要。

### 2.2 为什么调整旧选择

只生成 Q/K/V 会保留 PMOCO 输出投影条件化的缺口，因此旧方案降为后续简化对照。

将一个共享条件矩阵放在 MoE 前后也可实现条件计算，但会额外限定所有专家共享同一个左乘或右乘因子，并引入放置顺序的选择。方阵时不必降秩；问题是跨专家参数变化受到额外绑定。直接生成各专家参数，是对现有输出位置更直接的融合，没有实验依据把共享因子方案当成更优默认。

### 2.3 规模与执行

默认 embedding=128、heads=8、head_dim=16、experts=4、Top2。

| 对象 | 形状 |
| --- | --- |
| Encoder 节点编码 | `[B,N+1,128]` |
| 动态属性 | `[B,P,6]` |
| Query 输入 | `[B,P,134]` |
| Wq(lambda) | `[128,134]` |
| Wk(lambda)、Wv(lambda) | 各 `[128,128]` |
| 四专家输出矩阵 | `[4,128,128]` |
| 四专家 bias | `[4,128]` |
| MoE 输出 | `[B,P,128]` |
| logits / mask / probability | `[B,P,N+1]` |

一个偏好生成 115,968 个参数值，FP32 约 0.44 MiB，同 batch/POMO 共用，整条序列只生成一次。此数字不是整模型训练显存估计；实际显存还包括 attention、rollout 和反向图。

保留原 SparseDispatcher、noisy top-k 和负载均衡计算。Decoder 专家通过 functional linear 使用生成 Tensor，保持到超网络的梯度；不能通过 `.data`、`load_state_dict` 或新建 Parameter 注入权重。Encoder 的原 MLP 专家保持普通可训练参数。

## 3. 任务、目标和状态

### 3.1 保持原多任务问题族

仍使用原六类训练任务：`CVRP, OVRP, VRPB, VRPL, VRPTW, OVRPTW`。原十六环境均支持同一多目标接口，另十类保留未见约束组合的评价含义。沿用原实例生成和读取，客户特征保持 `(x,y,demand,tw_start,tw_end)`；不额外构造第二套坐标或成本矩阵。

原 task information 通过特征、动态状态与 mask 体现，不默认启用原代码未实际使用的 `prob_emb`，也不增加任务 ID embedding。

### 3.2 目标定义

\[
f_{\rm total}=\sum_{r\in\mathcal R}d_t(r),\qquad
f_{\rm max}=\max_{r\in\mathcal R}d_t(r).
\]

两者都按有效行驶距离最小化。闭合路线计回仓边；开放路线不计回仓分隔边。等待和服务时间仍用于时间窗约束，不计入这两个目标。Backhaul、容量、路线长度上限和时间窗的状态转移及 mask 保持原含义。

共享路线账本同时服务目标与新增状态。逐步计算有效边长，沿用 `loc_scaler` 的逐边舍入约定，累计：

- T_t：已走过全部有效边的累计总距离，包含当前路线。
- ell_t：当前车辆路线的有效累计距离，内部账本。
- M_t：截至当前所有已完成路线与当前路线前缀的最大距离。

每走一条边，先更新 T、ell、M；回到 depot 时再清零 ell。开放任务的回仓分隔边成本为零。连续 depot、完成轨迹的 padding 都不能增加目标。终局直接得到 `f_total=T`、`f_max=M`。

原物理约束状态与新目标账本分开维护。尤其不能为了计算新目标，改变原时间窗或路线限制的含义。第一目标须与原 MVMoE 同路线的成本核对一致。

### 3.3 Decoder 补充累计目标信息

\[
a_t=(load,time,length,open,T_t/s_1,M_t/s_2).
\]

保留原四属性，再加入两个确定性统计。原因是 Tchebycheff 是非加性的：最终瓶颈取决于已累计总距离以及历史最长路线，仅凭当前路线信息不足以区分这类目标进度。加入这两项不需要 RNN、历史 encoder 或新 critic。

这里补充的是新目标直接依赖的累计信息，不据此声称已证明所有原任务的状态表示具有严格的充分性。

## 4. Hypernetwork：优先采用论文的非线性映射

采用：

```text
lambda[2]
  → Linear(2,128) → ReLU
  → Linear(128,128) → ReLU
  → Linear(128,14)
  → 7组，每组2维：Q、K、V、O1、O2、O3、O4

Q 组 → 无bias生成头 → Wq[128,134]
K/V组 → 各自无bias生成头 → Wk/Wv[128,128]
每个O组 → 独立无bias生成头 → 16384个W值 + 128个b值
```

两个 128 维隐藏层和 ReLU 来自 PMOCO 论文；分组 latent 和矩阵生成头复用其当前代码的参数压缩思路，每组二维是代码来源设置。输出为14维及每专家生成头，是本次融合的尺寸组织。

PMOCO 当前仓库的连续 Linear(2,256)→Linear(256,256)→Linear(256,8) 没有激活，参数生成整体退化为偏好的仿射函数。合理性优先时，不把这一与论文不同的限制冻结为 backbone 默认。

从头训练，沿用标准 Linear 初始化。单目标 checkpoint 不作为融合模型的断点；以后若做部分初始化，需独立记录映射和新增参数，不能假定 strict load 可用。

## 5. Tchebycheff 的尺度、偏好与损失

### 5.1 固定单位校准，不增加学习模块

旧 raw TCH 中，`f_total >= f_max` 导致整个 `lambda_total >= lambda_max` 半区恒退化为总距离优化。保留 TCH 方法，但用固定正尺度定义偏好坐标：

\[
\tilde f_j=f_j/s_j,\quad s_j>0,\qquad
g_\lambda(F)=\max_j\lambda_j\tilde f_j.
\]

零 ideal 保留为距离的合法下界。尺度不冒充真实 ideal/nadir，也不是 Pareto 范围估计。正比例变换不改变支配关系；上式相当于在原目标上使用权重 `lambda_j/s_j`。它改变偏好的坐标解释，不保证前沿均匀，也不保证每个偏好对应不同解。

尺度采用一套明确、可重现的初始化校准：

1. 每个训练配置固定校准 seed，仅从六个训练任务各生成等量实例；工程默认每任务256个，数量与seed作为本工作区配置记录。
2. 使用已下载的原 MVMoE/4E 单目标参考 checkpoint，原 argmax/POMO、无增强，按总距离选出每实例的一条可行方案；不额外训练参考模型。
3. 用正确的统一路线账本计算这条方案的两个目标。
4. 六任务等权取均值：`s_j=(1/6)*sum_task(mean_instance(f_j))`，检查有限且严格为正后冻结。
5. 全十六任务共享同一对尺度。十个未见任务不生成校准实例，不按验证/测试目标重新估计；部署到未见规模也不在线重估。

主配置N=100时参考 checkpoint 为 `Routing-MVMoE/pretrained/mvmoe_4e_n100/epoch-5000.pt`。记录 checkpoint hash、实例生成配置、校准seed、逐任务统计和最终尺度。校准使用独立随机状态并恢复训练RNG，避免改变训练序列。参考策略只决定单位，未作为教师或训练标签。

这项校准是本次融合的显式工程约定，不是 PMOCO 原训练代码已有步骤。它保留原训练/评价任务划分，不利用未见任务统计作分布适配。

### 5.2 偏好采样遵守论文定义

双目标默认 `u~Uniform(0,1); lambda=(u,1-u)`，等价于 Dirichlet(1,1)。每 batch 一个任务、一个偏好，共享于全部实例及POMO组。

原代码 `rand(2)/sum` 偏向中间，不等于论文的均匀单纯形分布，因此不再作为主方案默认。训练不另加端点混合比例；测试沿用含端点的101个偏好，记录端点实际表现，不预设其单目标最优性。

### 5.3 回报单位与原 MoE 损失对接

为避免目标改用无量纲单位后，机械照抄原辅助损失造成无意的相对尺度变化，采用：

\[
R=-s_1 g_\lambda(F)
=-\max(\lambda_1f_1,\lambda_2(s_1/s_2)f_2).
\]

其中 s1 是已冻结的总距离尺度，没有新增可调系数。候选排序与 g 完全一致；当 `lambda=(1,0)` 时精确恢复 `R=-f_total` 的原回报单位。恢复的是回报形式，模型本身仍是融合后的模型。

\[
A_{b,p}=R_{b,p}-\operatorname{mean}_{p'}R_{b,p'},\qquad
L=\operatorname{mean}[-A\log\pi]+L_{\rm MoE}.
\]

同实例、同偏好的 POMO baseline 以及 MoE 均衡损失形式保持。原代码实际系数 `1e-3` 作为显式起始配置，不重复加权。检查主损失与辅助损失实际量级；量纲锚定不保证该系数已最优，也不引入自动梯度平衡算法。

## 6. 训练和解码合同

每 batch：采原训练任务 → 生成实例 → 采均匀偏好 → reset环境与目标账本 → hypernetwork生成全部decoder投影与专家参数 → encoder与K/V缓存 → 原POMO解码 → 目标向量/标量回报 → 原REINFORCE与MoE损失 → 更新。

`assign(lambda)` 必须在该偏好的 K/V 计算之前；切换偏好时同时重算Q/K/V和所有输出专家参数。生成Tensor保持autograd，只按一个偏好生成一次，不对每条轨迹复制整套矩阵。

Encoder保持偏好无关，因此在eval/no_grad下同实例的编码可跨偏好复用；decoder K/V一定重算，训练时也不复用旧计算图。可先验证逐偏好完整重算与缓存编码的结果一致，再启用该已有共享encoder性质带来的优化。

主配置继续使用原 MVMoE 已有设置：embedding128、六encoder层、8×16 heads、FFN512、4 experts/Top2、N=POMO=100、train batch128、Adam lr1e-4、weight_decay1e-6、seed2023。编码器与超网络等均正常训练，不冻结encoder。

原训练总预算作为来源配置保留，不因设计稿存在就启动5000 epochs。checkpoint保存完整模型/优化器/调度器、RNG、尺度buffer、校准记录、目标定义、任务集合、上游提交和实际配置。

## 7. 目标实现与评价正确性

继续遵守 PMOCO 论文的最长路线距离定义，纠正原仓库分段边界错误：

```text
depot=(0,0), a=(1,0), b=(0,2), c=(0,3)
0→a→0→b→c→0
闭合路线：2和6，正确目标(8,6)，原PMOCO函数实际返回(8,4)。
对应开放路线正确目标：(4,3)。
```

上述原函数结果已作确定性核验；当前仍未修改原函数。新实现应保留对照与出处，不复制已知错误。

推理在同一实例上扫描101个偏好。每实例每偏好按同一g选一条真实候选，再以同一索引读取完整目标和路线；不分别优化两个分量后拼点。端点scalar精确同分时按另一目标破平，内部偏好的精确同分按预先固定的二目标字典序处理；随后仍逐实例作非支配过滤。

保留 `objective_values[B,101,2]`、preferences、真实路线、非支配mask、task/instance/checkpoint/seed/augmentation/scale元数据。原始距离用于展示与可行性检查，不能把actor回报标为总距离。

HV逐实例计算后汇总；不复制原demo的“跨实例平均点集再算HV”。训练zero ideal、训练单位尺度、HV正参考点分别记录。

单位方形合成实例的工程检查可使用测试前固定的保守参考点：单边上界dmax（考虑舍入），总距离≤2N*dmax、最长路线≤(N+1)*dmax，参考点略取大于两上界。这个点有效但区分度可能弱；正式研究比较需预先声明更有区分度的共享参考点和超框处理，不能看到测试结果后调整。PMOCO原CVRP参考点只用于其相应设置的兼容记录，不自动推广到十六任务。

## 8. 代码落点与验收

原两个仓库继续保留为来源对照；实现落在独立 `mtmo/`，保留许可证与适配说明：

```text
mtmo/models/       MVMoE encoder + preference hypernetwork + conditional decoder MoE
mtmo/envs/         原环境包装 + 统一有效边账本/路线目标
mtmo/calibration/  固定训练任务单位校准
mtmo/trainer.py    POMO、偏好与标量训练回报
mtmo/tester.py     逐实例偏好解集与评价
configs/          当前模型/任务/尺度来源与训练参数
```

验收聚焦backbone正确性：

1. 闭合/开放手工路线及padding/路线重排正确；第一目标与原环境一致；账本前缀与终局解析一致。
2. Query134维，Q/K/V与各专家W/b形状正确；未选动作mask概率为零；所有生成头能获得梯度，原gate与encoder也能更新。
3. 固定偏好时专家参数在整条rollout中一致；换偏好后不会复用旧专家或K/V。
4. 六训练任务都可完成前向/反向/更新；十六环境都可合法构造并返回二目标。
5. 单目标端点的回报形式与原总距离回报一致；尺度和HV单位变换不改变其应有排序/支配关系。
6. checkpoint恢复包含尺度与校准来源；结果始终绑定真实方案；按实例计算前沿。

不以“每个偏好生成不同解”“超过PMOCO的HV”“未见组合泛化成立”作为工程完成条件。性能与泛化属于后续训练证据。

## 9. 主要依据

- [MVMoE，ICML 2024](https://proceedings.mlr.press/v235/zhou24c.html)：统一VRP状态、MoE与训练组织。
- [PMOCO，ICLR 2022](https://arxiv.org/abs/2203.15386)：§3–4条件参数策略/TCH/偏好分布；附录模型和MOCVRP定义。
- `papers/PMOCO/PMOCO_latex_source/iclr2022_conference_arxiv.tex:248`：均匀偏好；`:520`：128/ReLU；`:611`：双目标距离；`:759`：MOCVRP目标尺度差异。
- `Routing-MVMoE/models/MOEModel.py:255`：QKV和MoE输出位置；`:76`：原动态属性。
- `Routing-MVMoE/models/MOELayer.py:182`：仿射专家；`:350`：实际辅助系数与gate执行。
- `PMOCO/MOCVRP/POMO/MOCVRPModel.py:172`：当前超网络及分组生成头；`:193`：assign。
- `PMOCO/MOCVRP/POMO/MOCVRPTrainer.py:150`：当前代码偏好分布；`:176`：raw TCH。
- `PMOCO/MOCVRP/POMO/MOCVRPEnv.py:198`：原最长路线计算；`MOCVRPTester.py:166`：原均值结果接口。

当前方案的核心是对原计算位置作完整条件化、为新目标提供累计状态，并明确目标单位；没有另加问题族、任务embedding、额外router、LoRA、搜索器或新的优化目标。
