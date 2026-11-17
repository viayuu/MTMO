# MTMO backbone v0 设计方案

> 本文是旧草案。2026-11-18 起以 [当前 backbone 方案](MTMO_BACKBONE_PLAN.md) 为准。旧稿将改动量放得过重；Q/K/V-only、四维动态状态、raw TCH、无激活超网络和原代码偏好采样等选择均已重新审视，不再作为当前默认。

日期：2026-11-17。状态：设计稿，尚未实现或启动训练。

目标：以原 MVMoE 为主体，迁入 PMOCO 的 MOCVRP 多目标定义、偏好超网络和训练机制，形成一条可训练、可解码、可保存结果的最小多任务多目标通路。此阶段的验收是实现正确与接口完整，不要求性能超过原方法。

## 1. 设计边界与来源

固定上游版本：

- MVMoE：`af29e5af0595f94f3ecc3bc46d72df1089a62682`。
- PMOCO：`8e734c819b0adaa2851c51ac800df5c5c7d5bbb6`。

优先继承当前仓库实际执行路径；问题目标优先遵守论文的数学定义。发现代码与论文不一致时逐项记录，避免混用两套设置。

| 部分 | v0 决定 | 来源与改动性质 |
| --- | --- | --- |
| 主干 | 原版 MVMoE/4E，显式 `model_type=MOE` | 原论文、原仓库已有模型配置；仓库 CLI 默认其实是 `MOE_LIGHT`，本方案明确选主模型调通一条通路 |
| 编码器、专家与 gate | 保留六层编码器、原 MoE、4 experts、Top2、node/input-choice routing | MVMoE 原实现 |
| 任务与实例 | 原六类训练任务，原十六类环境，原数据生成与读取 | MVMoE 原实现 |
| 双目标 | 全部车辆总路线距离、最长单车路线距离 | PMOCO MOCVRP 论文定义 |
| 偏好条件化 | 超网络生成 decoder Q/K/V；输出层仍是原 MVMoE MoE | 两套模型相遇处的最小接口适配 |
| 偏好采样与标量化 | 每 batch 一个偏好，原 `rand → normalize`，原 MOCVRP TCH，训练参考点零 | PMOCO MOCVRP 原代码 |
| 优化 | POMO REINFORCE + 原 MoE 辅助损失 | 两仓库共同范式，保留 MVMoE 主训练流程 |
| 结果 | 每实例、每偏好的真实解及完整目标向量 | 为多目标输出适配原单标量接口 |

本阶段不增加任务 embedding、偏好 router、条件专家权重生成、LoRA、新损失、目标归一化、偏好重采样、主动搜索或额外问题族。MOTSP/MOKP 属于 PMOCO 的其他任务，不纳入本次 VRP backbone。MOE_LIGHT 继续保留为原始参考，暂不并行实现另一套主配置。

## 2. 任务、目标与数据

### 2.1 任务集合

训练任务原样使用：`CVRP, OVRP, VRPB, VRPL, VRPTW, OVRPTW`。

全部环境继续包括原十六种变体。训练只采上述六种，十六环境均需实现相同的多目标接口；其余十类保留为原协议中的未见约束组合。这里的“支持十六环境”只表示接口和合法性可运行，不等于已证明零样本多目标泛化。

任务通过原有节点特征、动态属性与环境 mask 体现。MVMoE 代码虽然保留了 `prob_emb` 字段，但模型默认没有使用；v0 不把这个预留字段改成新的任务编码机制。

### 2.2 统一双目标

对任务 t 的一个可行解，按 depot 分隔得到车辆路线集合 R。令 d_t(r) 为一条路线在该任务中的有效行驶距离：

\[
f_1=\sum_{r\in R}d_t(r),\qquad f_2=\max_{r\in R}d_t(r).
\]

两个目标都最小化。第二目标统一命名为“最长单车路线距离”，避免将 PMOCO 借用的 makespan 一词误解为包含等待的工作时长。

| 任务属性 | 目标中如何计距离 | 保留的原环境规则 |
| --- | --- | --- |
| 闭合路线 | 出仓、客户间、回仓的边都计入该条路线 | 原容量、访问、终止规则 |
| O / open | 每条路线不计最后客户回仓的人工分隔边 | 原 open 状态与 mask |
| B / backhaul | 所有实际行驶的有效边计费 | 原配送/取货顺序、容量变化与 mask |
| L | 仍按行驶距离计算两目标 | 原 route limit 与 `length` 约束 |
| TW | 两目标不计等待和服务时间 | 原 `current_time`、等待、服务及时间窗可行性检查 |

沿用 MVMoE 的 `loc_scaler` 逐边舍入规则，两目标必须由同一套有效边长计算。连续 depot、已完成 rollout 的 depot padding 形成零长度空段。

无需构造第二套坐标、成本矩阵或新数据集。现有实例本身足以评价这两个路线结构目标。训练继续在线生成实例；现有固定数据文件继续用于相应检查与评价，不额外设计数据划分。

### 2.3 环境接口

建议在原环境外增加一个薄包装层，委托原环境的 `reset/pre_step/step/mask`，在终局从原始路线序列补充两个目标。原环境的单目标回报可保留用于核对；多目标训练使用新返回的目标向量。

```text
原环境终局：state, reward_total[B,P], done
包装层另提供：objectives[B,P,2]，按 [total_distance, max_route_distance] 排列
训练器：objectives + preference → scalar_reward[B,P]
```

包装层显式接收任务是否 open，不能靠模型猜测；不重写十六套 mask。路线目标计算集中到一个共享函数，避免各环境各写一份最长路线算法。

为了保持原模型，decoder 的四维动态输入仍为 `(load,current_time,length,open)`。最长已完成路线的历史值可在环境账本或终局路线解析中获得，v0 不把它新增到模型输入。这保留了原输入接口，但不能声称四维输入已是新 min-max 目标的充分状态；可能的表达限制留待后续证据判断。

## 3. 模型融合：保留 MVMoE 的 MoE，仅条件化 Q/K/V

### 3.1 不变部分

保留 depot 二维坐标输入、客户 `(x,y,demand,tw_start,tw_end)` 五维特征、原输入嵌入、六层编码器、编码器 MoE、InstanceNorm、残差结构、自回归节点选择、POMO 起点、mask 和最终 logit clipping。

编码器不接收偏好，符合 PMOCO 的共享编码思路。Decoder 仍接收当前节点编码和原四维动态属性。

### 3.2 必须明确的 combine 冲突

PMOCO MOCVRP 的超网络生成 Q/K/V 和单个 output-combine 投影。MVMoE 在这个 combine 位置已使用多个线性专家。

v0 的决定是：**保留 MVMoE 的原始 MoE combine，超网络只生成 Q/K/V。** 因此应称为“PMOCO 式偏好条件化的最小适配”，而非完整复制原 PMOCO decoder 的四个生成矩阵。

不在第一版为每个专家生成权重和 bias，也不额外叠加一个动态输出层。专家参数、gate、dispatcher 和辅助损失的训练方式保持原样，均参与正常训练。

这样偏好已经能经 Q/K/V 改变 attention 输出，并间接影响 decoder gate；无需额外加入 preference gate。编码器对同一实例的专家分配不随偏好改变，这是本版本的明确设计。

### 3.3 超网络具体配置

继承当前 PMOCO MOCVRP 仓库的生成器结构，移除 combine 生成分组：

```text
lambda[2]
  → Linear(2,256)
  → Linear(256,256)
  → Linear(256,6)
  → 分成 e_q[2], e_k[2], e_v[2]

e_q → Linear(2,128*132,bias=False) → Wq[128,132]
e_k → Linear(2,128*128,bias=False) → Wk[128,128]
e_v → Linear(2,128*128,bias=False) → Wv[128,128]
```

前三个 Linear 保留原代码的 bias；层间不自行添加激活函数；初始化沿用 `nn.Linear` 默认值。PMOCO 论文写的是 128 维隐藏层和 ReLU，而当前仓库实际是 256 维、无激活。本阶段明确沿用仓库实现，不能将二者混称。

`256→8` 改为 `256→6`、删除 combine 生成头，是只生成三个投影后的尺寸适配。Q 的输出量变大，是保留 MVMoE 四维动态上下文所必需。

所有生成矩阵按 PyTorch `F.linear` 的 `[out_features,in_features]` 排列，不能沿用 PMOCO 方阵下难以察觉的维度写法。

| 张量 | 形状 |
| --- | --- |
| 节点编码 | `[B,N+1,128]` |
| 当前节点编码 + 四维动态属性 | `[B,P,132]` |
| 生成的 Wq | `[128,132]` |
| 生成的 Wk / Wv | `[128,128]` |
| attention Q | `[B,8,P,16]` |
| attention K/V | `[B,8,N+1,16]` |
| 原 MoE combine 的输入/输出 | `[B,P,128]` |
| logits、mask、probabilities | `[B,P,N+1]` |

生成矩阵直接用于 functional linear，保留 autograd 连接；不用 `.data`、`load_state_dict` 或重新构造 Parameter 来注入这些动态权重。

## 4. 训练流程与固定默认

每 batch 的执行顺序：

```text
1. 从原六类任务均匀抽一个任务。
2. 用该任务原始生成器得到 B 个实例。
3. u = torch.rand(2); lambda = u / u.sum()。
4. 整个 batch 与全部 POMO rollout 共用该 lambda。
5. env.load_problems 后取得 env.reset() 的 reset_state；decoder.assign(lambda)，然后 model.pre_forward(reset_state) 编码并设置 K/V。
6. 按原 mask 完成 POMO 自回归构造。
7. 得到 F[B,P,2]，计算 TCH 标量回报。
8. 同实例、同偏好的 POMO 组内计算 baseline；反向更新全部参数。
```

采样方法按原代码命名为 `rand_normalized`，不称为严格均匀单纯形或 Dirichlet(1) 采样。每条完整构造序列内偏好和生成参数不变。

继承 PMOCO MOCVRP 的零参考点 TCH：

\[
g_\lambda(F)=\max(\lambda_1 f_1,\lambda_2 f_2),\quad R=-g_\lambda(F).
\]

\[
A_{b,p}=R_{b,p}-\operatorname{mean}_{p'}R_{b,p'},\qquad
L=\operatorname{mean}_{b,p}[-A_{b,p}\log\pi_{b,p}]+L_{\rm MoE}.
\]

不跨任务或跨偏好混算 baseline。按 MVMoE 原方式累积 encoder/decoder 的辅助损失，不额外乘第二次系数。当前 `MoE.forward` 的实际默认 `loss_coef=1e-3`；Trainer 注释中的 `1e-2` 不作为当前执行配置。

| 项目 | v0 主配置 |
| --- | --- |
| model_type | `MOE`，即原版 MVMoE/4E |
| embedding / encoder layers / heads / head dim | `128 / 6 / 8 / 16` |
| FFN / experts / Top-k | `512 / 4 / 2` |
| node size / POMO | `100 / 100` |
| train batch | `128` |
| Adam lr / weight decay | `1e-4 / 1e-6` |
| seed | `2023`，继承 MVMoE 默认 |
| 初始化 | 从头训练，原 Linear 默认初始化；不把单目标 checkpoint 当成多目标断点 |

MVMoE 的 5000 epochs、每 epoch 20000 实例、4501 epoch 学习率衰减可保留为来源参考配置；搭建阶段只做必要的小规模通路检查，不据此自动启动完整训练，也不把 PMOCO 的 200 epochs/100000 实例预算混入。

以后若需要部分预训练初始化，应单独定义参数映射。原 MVMoE `--checkpoint` 会 strict 加载并恢复 optimizer/epoch/scheduler，不能作为本版本的隐式 warm-start 开关。

## 5. 解码与结果接口

沿用 PMOCO 双目标的 101 个测试偏好：`[1-0.01*i,0.01*i]`，`i=0..100`，在同一批实例上逐个解码。

每个偏好下，先根据该偏好的 TCH 从 POMO 候选中选择一条真实方案，再用同一个候选索引读取两个目标和路线。保留 MVMoE 的 1/8 倍增强接口；增强开启时，在 POMO 与增强维联合选择并恢复为原实例路线。

建议输出：

```text
objective_values[B,101,2]
preferences[101,2]
selected_routes：每实例、每偏好的真实路线
nondominated_mask[B,101]
metadata：任务、实例来源、N、checkpoint、seed、增强设置、目标版本
```

第一版不优化编码缓存：像原 PMOCO 流程一样，每个偏好重新完成 `assign → pre_forward → rollout`，避免 stale K/V 或 MoE 辅助量污染。接口正确后再考虑复用 encoder；切换偏好时 K/V 必须重算。

不沿用原 MVMoE 的单目标 gap 作为多目标主指标，不跨不同偏好取一个 scalar “总最佳”，不分别取两个目标最小值拼成不存在的解。

backbone 必需输出为逐实例目标点、解与非支配集合。若计算 HV，应逐实例计算后汇总；参考点按任务和规模固定并报告，不能把训练 TCH 的零参考点拿来算 HV。PMOCO MOCVRP 的 n50 `(40,3)`、n100 `(60,3)` 只作为相应设置的来源记录，不未经检查推广到十六任务。

## 6. 已确认的原始实现差异与处理

### 6.1 最长路线边界错误：遵守论文定义

原 PMOCO MOCVRP 使用出边长度 cumsum 后在 depot 位置取值，可能把下一条路线的首段计入上一条。已通过未修改的原函数作确定性核验：

```text
depot=(0,0), a=(1,0), b=(0,2), c=(0,3)
路线序列：0 → a → 0 → b → c → 0
两条闭合路线长度：2 和 6
论文定义应为：F=(8,6)
当前原函数实际返回：F=(8,4)
```

v0 应按 depot 正确分段聚合有效边长，纠正这一边界计算；保留原源码供对照，并在适配实现中记录出处。开放版本同一例子的预期为 `(4,3)`。原 MVMoE 的第一目标计算仍须逐实例核对一致。

### 6.2 TCH 的偏好退化：保留并记录

因为 `f1 >= f2`，当 `lambda1 >= lambda2` 时必有：

\[
\max(\lambda_1 f_1,\lambda_2 f_2)=\lambda_1 f_1.
\]

这是继承原 raw TCH / z=0 后就有的性质，意味着至少半个偏好区间的标量目标退化为总路程优化。v0 不新增目标归一化或偏好再分配；记录重复解与非支配点数，不要求 101 个偏好产生 101 个不同解，也不宣称均匀覆盖真实前沿。

### 6.3 PMOCO 原测试的均值前沿

原 MOCVRP Tester 返回跨实例平均目标，外层再对这些均值点算 HV。这与“每个实例的 HV 再取平均”不同。v0 必须保留逐实例数据；平均目标曲线可用于兼容展示，但不能将其 HV 当成逐实例平均 HV。此处属于结果接口与评价口径的明确适配。

## 7. 建议的最小代码落点

原 `Routing-MVMoE/` 和 `PMOCO/` 保留为上游对照。后续实现放在独立的 `mtmo/`，只迁入所需文件并保留原许可证、来源和改动说明，不先对整个工程做风格或结构重构。

```text
mtmo/
  models/           # MVMoE/4E 编码器、MoELayer与适配的偏好解码器
  envs/             # 薄环境包装与统一路线双目标计算
  trainer.py        # 原训练流程 + preference/TCH
  tester.py         # 每偏好解码与逐实例结果
configs/
  backbone_v0.yaml  # 本文明确的主配置
train_mtmo.py
test_mtmo.py
```

环境包装复用原环境实现；不复制或维护十六套新的约束逻辑。工程导入与路径处理只做让新入口可运行所需的适配。checkpoint 记录融合版本、上游提交、完整配置、目标定义、偏好采样方式、模型/optimizer/scheduler状态及恢复所需的随机状态，保证后续能识别模型实际训练设置。

## 8. 实现顺序与完成条件

| 顺序 | 交付 | 必须检查的内容 |
| --- | --- | --- |
| 1 | 路线双目标函数和环境包装 | 闭合例 `(8,6)`、开放例 `(4,3)`；调换路线顺序和添加 depot padding 后目标不变；f1 与原 MVMoE 一致 |
| 2 | 偏好 Q/K/V 解码器 | 非方形 Q 尺寸正确；mask 概率合法；生成矩阵保留梯度；decoder MoE 仍被实际调用 |
| 3 | 单任务通路 | CVRP 能完成 rollout、双目标计算、TCH、backward、optimizer step、save/load |
| 4 | 多任务通路 | 六训练环境均可更新；十六环境均可构造并输出目标；任务/偏好不跨组混算 baseline |
| 5 | 偏好扫描与结果 | 保存真实路线和对应二目标；非支配筛选逐实例进行；切换偏好没有复用旧 K/V |

这些是后续实现时需要执行的验收，当前尚未完成。小规模调试配置须与完整训练配置分别记录，不将通路检查称为训练结果。

不把“偏好改变必须产生不同路线”“HV 超过 PMOCO”“未见任务泛化成立”作为 backbone 完工门槛。这些属于后续训练和研究验证。

## 9. 可追溯依据

- [MVMoE 官方论文](https://proceedings.mlr.press/v235/zhou24c.html)，方法部分及训练设置。
- [PMOCO 官方论文](https://arxiv.org/abs/2203.15386)，§3–4、附录模型设置与 MOCVRP 定义。
- `Routing-MVMoE/train.py:34` 起：模型、训练与优化参数；`utils.py:94`：原任务集合。
- `Routing-MVMoE/models/MOEModel.py:28`：编码输入；`:76`：四维动态上下文；`:255`：Q/K/V 与 MoE combine。
- `Routing-MVMoE/models/MOELayer.py:182`：专家结构；`:350`：实际辅助损失默认系数。
- `Routing-MVMoE/Trainer.py:101`：任务采样；`:119`：rollout；`:138`：REINFORCE。
- `Routing-MVMoE/envs/CVRPEnv.py:265` 与 `OVRPEnv.py:265`：闭合/开放距离与舍入规则。
- `PMOCO/MOCVRP/POMO/MOCVRPModel.py:172`：超网络；`:193`：assign。
- `PMOCO/MOCVRP/POMO/MOCVRPTrainer.py:150`：偏好；`:176`：TCH；`:190`：baseline。
- `PMOCO/MOCVRP/POMO/MOCVRPEnv.py:198`：原目标计算及边界问题。
- `PMOCO/MOCVRP/POMO/MOCVRPTester.py:166` 与 `test_mocvrp_n100.py:107`：均值输出及扫描/HV。

本方案的必要适配清单：Q 的动态输入尺寸；Q/K/V 与 MoE combine 的接口取舍；路线最长距离的正确分段；标量回报与向量结果的分离；逐实例多目标输出。其余部分优先沿用原实现。
