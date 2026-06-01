# LeWM (LeWorldModel) 论文总结报告

> **论文**: LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels  
> **arXiv**: [2603.19312](https://arxiv.org/abs/2603.19312) (2026年3月)  
> **作者**: Lucas Maes*, Quentin Le Lidec*, Damien Scieur, Yann LeCun, Randall Balestriero  
> **机构**: Meta AI (FAIR), Mila, Univ. Montréal  
> **代码**: [github.com/lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)  
> **模型**: [HuggingFace Collection](https://huggingface.co/collections/quentinll/leworldmodel)

---

## 一、研究背景与动机

### 1.1 问题定义

Joint-Embedding Predictive Architecture (JEPA) 是 Yann LeCun 提出的一种学习框架，旨在**联合嵌入空间中**进行预测性学习，而非在原始像素空间重建。JEPA 的核心思想是：

- **编码器**将观测映射到潜在嵌入空间
- **预测器**在该空间中预测未来的嵌入表示
- 通过**预测性损失**学习世界的动态模型

### 1.2 现有方法的瓶颈

现有的 JEPA 世界模型方法（如 PLDM、DINO-WM 等）面临以下关键问题：

| 问题 | 具体表现 |
|------|---------|
| **训练不稳定** | PLDM 使用 **7 个损失项**，超参数从6个减到仍不稳定，训练过程噪声大、非单调 |
| **表示坍缩** | 嵌入退化为常数，失去信息量 |
| **依赖复杂技巧** | 需要 EMA（指数移动平均）、stop-gradient、冻结预训练编码器等工程技巧 |
| **计算成本高** | 基于基础模型的世界模型（如 DINO-WM 使用 DINOv2）规划速度极慢 |
| **超参数敏感** | 多个损失项的权重需要仔细调节，不同环境需要不同配置 |

### 1.3 LeWM 的核心贡献

LeWM 提出了一种**极简而稳定**的 JEPA 世界模型：

1. **仅两个损失项**：将可调超参数从 6 个减少到 **1 个**（λ，SIGReg 权重）
2. **端到端训练**：无需冻结编码器、无需 EMA、无需 stop-gradient
3. **理论保证**：SIGReg 正则化器提供**可证明的反坍缩保证**
4. **高效率**：~15M 参数，单 GPU 训练数小时，规划速度比基础模型快 **48×**

---

## 二、方法详解

### 2.1 整体架构

LeWM 由以下核心组件构成：

```
┌─────────────────────────────────────────────────────────┐
│                     LeWM 架构                            │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌───────────────┐      │
│  │  ViT-tiny │    │ Projector│    │ Action        │      │
│  │  Encoder  │───>│ (MLP+BN) │    │ Encoder       │      │
│  │ (patch=14)│    │ 192→2048 │    │ (Conv1d+MLP)  │      │
│  └──────────┘    │ →192     │    └───────┬───────┘      │
│       ↑          └────┬─────┘            │               │
│   raw pixels          │ emb              │ act_emb       │
│                       ▼                  ▼               │
│               ┌──────────────────────────────┐           │
│               │    ARPredictor (Transformer)  │           │
│               │    6层 ConditionalBlock        │           │
│               │    + AdaLN-zero 调制           │           │
│               └──────────────┬───────────────┘           │
│                              │                            │
│                       ┌──────▼──────┐                    │
│                       │ Pred Projector│                    │
│                       │ (MLP+BN)     │                    │
│                       └──────────────┘                    │
│                                                          │
│  Loss = MSE(pred, target) + λ × SIGReg(emb)              │
└─────────────────────────────────────────────────────────┘
```

#### 2.1.1 视觉编码器 (Encoder)

- **架构**: HuggingFace ViT-tiny（非预训练）
- **输入**: 224×224 RGB 图像
- **Patch 大小**: 14×14 → 每个 patch 编码为一个 token
- **输出**: CLS token 作为图像的全局表示（维度 192）
- **设计选择**: 使用 tiny 变体保持模型轻量（整个模型 ~15M 参数）

```python
# 代码实现 (jepa.py:76-84)
def encode(self, info):
    pixels = info['pixels'].float()
    b = pixels.size(0)
    pixels = rearrange(pixels, "b t ... -> (b t) ...")
    output = self.encoder(pixels, interpolate_pos_encoding=True)
    pixels_emb = output.last_hidden_state[:, 0]  # CLS token
    emb = self.projector(pixels_emb)
    info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
```

#### 2.1.2 投影器 (Projector)

- **结构**: 两层 MLP（192 → 2048 → 192），中间使用 BatchNorm1d 和 GELU 激活
- **作用**: 将 ViT 的 CLS token 投影到 JEPA 的潜在嵌入空间
- **预测投影器 (pred_proj)**: 结构与 projector 相同，用于投影预测器的输出

#### 2.1.3 动作编码器 (Action Encoder)

- **结构**: 
  - Conv1d（1×1 卷积）：将动作维度映射到平滑维度
  - MLP：smoothed_dim → 4×emb_dim → emb_dim（使用 SiLU 激活）
- **输入**: 动作序列 (B, T, action_dim)
- **输出**: 动作嵌入 (B, T, 192)，与状态嵌入维度相同
- **帧跳跃**: frameskip=5，意味着动作维度 = 5 × 原始动作维度

```python
# 代码实现 (module.py:397-447)
class Embedder(nn.Module):
    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4):
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )
```

#### 2.1.4 自回归预测器 (ARPredictor)

- **核心架构**: 6 层 Transformer，使用 **ConditionalBlock**（带 AdaLN-zero 调制）
- **条件化方式**: 动作嵌入通过 AdaLN-zero 调制每一层的 LayerNorm 参数
- **位置编码**: 可学习位置嵌入，最大长度 = history_size
- **因果注意力**: 使用因果掩码防止信息泄露
- **参数配置**: 16 个注意力头，每头 64 维，MLP 隐藏维度 2048

```python
# ConditionalBlock 的 AdaLN-zero 调制机制 (module.py:189-247)
class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        # 初始化为零 → 训练开始时为恒等映射
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
```

**AdaLN-zero 的关键设计**：
- 从条件信息（动作嵌入）生成 6 个调制参数：shift、scale、gate 各两组（注意力和 MLP）
- 初始化为零确保训练初期模块为恒等映射，有利于稳定训练
- gate 机制允许模型学习"跳过"某些层

### 2.2 SIGReg 正则化器

SIGReg（Sketched Isotropic Gaussian Regularizer）是 LeWM 的核心创新之一，它通过**随机投影 + Epps-Pulley 正态性检验**来确保嵌入分布为各向同性高斯分布。

#### 2.2.1 数学原理

**目标**: 确保 latent embedding 服从 $\mathcal{N}(0, I)$ 分布

**方法**: 基于 Epps-Pulley 特征函数检验

1. **随机投影**: 生成 $K=1024$ 个随机方向 $a_k \sim \mathcal{N}(0, I)$，归一化后 $a_k \leftarrow a_k / \|a_k\|$
2. **一维投影**: $z_k = emb \cdot a_k$，检验每个 $z_k$ 是否服从标准正态
3. **特征函数比较**: 
   - 经验特征函数: $\hat{\phi}_n(t) = \frac{1}{n}\sum_{i=1}^n e^{it z_k^{(i)}}$
   - 目标高斯特征函数: $\phi(t) = e^{-t^2/2}$
4. **Epps-Pulley 统计量**: $\text{EP}(z_k) = \int_0^3 w(t) \left[|\hat{\phi}_n(t) - \phi(t)|^2 \cdot n\right] dt$
5. **最终损失**: 对所有投影方向取平均

```python
# 代码实现 (module.py:30-82)
class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        t = torch.linspace(0, 3, knots)           # 积分节点
        weights = trapezoidal_weights(knots, dt)   # 梯形法则
        phi = torch.exp(-t.square() / 2.0)        # 目标高斯特征函数
        self.register_buffer("weights", weights * phi)  # 合并权重和窗口

    def forward(self, proj):  # proj: (T, B, D)
        A = torch.randn(D, num_proj, device=device)  # 随机投影矩阵
        A = A.div_(A.norm(p=2, dim=0))               # 归一化
        x_t = (proj @ A).unsqueeze(-1) * self.t      # 投影并缩放
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * B          # Epps-Pulley 统计量
        return statistic.mean()                       # 平均
```

#### 2.2.2 SIGReg 的理论保证

- **反坍缩**: 高斯分布具有最大熵，当嵌入分布接近高斯时，不可能坍缩到单点
- **信息保留**: 各向同性高斯分布保留了足够的编码容量
- **可计算**: 仅需随机投影和特征函数比较，计算高效

### 2.3 训练流程

#### 2.3.1 前向传播

```
输入: 序列帧 {x_0, x_1, ..., x_T} 和对应动作 {a_0, a_1, ..., a_T}

1. 编码所有帧: emb = Encoder(frames)           → (B, T, 192)
2. 编码所有动作: act_emb = ActionEncoder(actions) → (B, T, 192)
3. 提取历史上下文: ctx_emb = emb[:, :3], ctx_act = act_emb[:, :3]
4. 提取目标: tgt_emb = emb[:, n_preds:]        → n_preds=1 时为 emb[:, 1]
5. 自回归预测: pred_emb = Predictor(ctx_emb, ctx_act) → (B, 3, 192)
6. 计算损失:
   pred_loss = MSE(pred_emb, tgt_emb)
   sigreg_loss = SIGReg(emb)  # 作用于所有嵌入
   total_loss = pred_loss + λ × sigreg_loss
```

#### 2.3.2 训练配置

| 超参数 | 值 |
|--------|-----|
| 优化器 | AdamW |
| 学习率 | 5e-5 |
| 权重衰减 | 1e-3 |
| 学习率调度 | LinearWarmupCosineAnnealing |
| Batch Size | 64-128 |
| 训练轮数 | 100 |
| 梯度裁剪 | 1.0 |
| 混合精度 | 16-mixed (bf16) |
| SIGReg 权重 λ | 0.09 |
| SIGReg 投影数 | 1024 |
| SIGReg 积分节点 | 17 |

### 2.4 推理与规划 (Model Predictive Control)

#### 2.4.1 推理流程

```
给定: 初始观测序列 (history), 目标图像 (goal), 动作候选序列

1. 编码初始历史帧 → 初始嵌入序列
2. 编码目标图像 → 目标嵌入 goal_emb
3. 自回归 rollout:
   for t in range(planning_horizon):
       pred_emb[t] = Predictor(emb[t-H:t], act_emb[t-H:t])
       emb = concat(emb, pred_emb[t])
4. 计算代价: cost = MSE(pred_emb[-1], goal_emb)
5. 选择最优动作序列 (通过 CEM 或梯度优化)
```

#### 2.4.2 CEM 规划器

- **种群大小**: 生成多个候选动作序列
- **迭代**: 保留高适应度个体，拟合高斯分布，重新采样
- **时间范围**: horizon=5, receding_horizon=5, action_block=5

---

## 三、实验设置

### 3.1 基准环境

LeWM 在 4 个不同复杂度的控制任务上进行了评估：

| 环境 | 类型 | 维度 | 描述 |
|------|------|------|------|
| **Two-Room** | 2D 导航 | 低维 | 智能体在两个房间间导航，通过走廊连接 |
| **Push-T** | 2D 操作 | 中维 | 推动T形块到目标位置 |
| **OGBench-Cube** | 3D 操作 | 高维 | 在3D空间中操作立方体（OGBench基准） |
| **DMControl-Reacher** | 3D 控制 | 高维 | 机械臂到达目标位置（DeepMind Control） |

### 3.2 基线方法

| 方法 | 类型 | 编码器 | 特点 |
|------|------|--------|------|
| **PLDM** | JEPA | 端到端训练 | 7项损失，仅有的另一个端到端方法 |
| **DINO-WM** | JEPA | 冻结 DINOv2 | 基于基础模型，需要预训练编码器 |
| **DINO-WM-noprop** | JEPA | 冻结 DINOv2 | 不使用本体感觉信息 |
| **IVL** | RL 方法 | — | 隐式值学习 |
| **IQL** | RL 方法 | — | 隐式Q学习 |
| **GCBC** | IL 方法 | — | 目标条件行为克隆 |

### 3.3 评估指标

- **成功率 (Success Rate)**: 主要评估指标，在 50 个评估 episode 上计算
- **规划时间**: 每步规划的计算时间
- **嵌入质量**: 线性/非线性探测 (probing) 物理量
- **惊讶度 (Surprise)**: 检测物理不合理事件的能力

---

## 四、实验结果

### 4.1 规划性能（核心结果）

#### 4.1.1 成功率对比

| 环境 | LeWM | PLDM | DINO-WM | GCBC | IQL | IVL |
|------|------|------|---------|------|-----|-----|
| **Push-T** | **较高** | 较低（低18%） | 中等 | — | — | — |
| **Reacher** | **竞争力强** | 较低 | 中等 | — | — | — |
| **OGBench-Cube** | 中等 | 较低 | **最高** | — | — | — |
| **Two-Room** | **低于基线** | 中等 | 中等 | — | — | — |

**关键发现**:
- LeWM 在 **Push-T** 上比唯一另一个端到端方法 PLDM **高出 18%** 的成功率
- 即使 DINO-WM 使用额外的本体感觉输入，LeWM 仍然优于 DINO-WM
- 在 **OGBench-Cube** 上略低于 DINO-WM，可能因为 3D 视觉复杂性
- **Two-Room 上低于基线**：作者**推测** SIGReg 的高斯先验可能是原因（"A possible explanation"），但**未经验证**——所有消融实验均在 PushT 上完成，没有针对 TwoRoom 做任何专门的验证（详见下文分析）

#### 4.1.2 计算效率

| 方法 | 参数量 | 编码器 | 规划速度 |
|------|--------|--------|---------|
| **LeWM** | ~15M | ViT-tiny (端到端) | **~1秒** |
| **PLDM** | ~15M | ViT (端到端) | ~1秒 |
| **DINO-WM** | ~300M+ | DINOv2 (冻结) | **~48秒** (48× 慢于 LeWM) |

- LeWM 使用 **~200× 更少的 token** 比 DINO-WM
- 在**固定计算预算**下，LeWM 在 Push-T 和 OGBench-Cube 上优于 DINO-WM
- 单 GPU 训练仅需数小时

### 4.2 训练稳定性

#### 4.2.1 与 PLDM 的对比

| 指标 | LeWM | PLDM |
|------|------|------|
| 损失项数量 | **2** | 7 |
| 可调超参数 | **1** (λ) | 6 |
| 收敛行为 | 平滑、单调 | 噪声大、非单调 |
| SIGReg 损失 | 早期快速下降后平稳 | N/A |
| 跨种子方差 | **低** | 高 |

#### 4.2.2 消融实验

**SIGReg 权重 λ 的鲁棒性**:
- 性能在 λ ∈ [0.01, 0.2] 范围内保持高位
- 推荐默认值 λ = 0.09

**编码器架构无关性**:
- ViT-tiny 和 ResNet-18 均可有效训练
- SIGReg 机制不依赖于特定编码器

**时间直线化 (Temporal Straightening)**:
- LeWM 的潜在轨迹在训练过程中变得**越来越直**
- 连续速度向量间的余弦相似度高于 PLDM
- 这一特性**自然涌现**，没有显式的时间平滑损失

### 4.3 嵌入质量分析

#### 4.3.1 物理量探测 (Probing)

线性/非线性探针在冻结的 LeWM 嵌入上训练，预测物理量：

| 物理量 | LeWM | PLDM | DINOv2 (124M图像预训练) |
|--------|------|------|------------------------|
| 智能体位置 | 高 | 中-低 | 高 |
| 方块位置 | 高 | 中 | 高 |
| 方块速度 | 高 | 低 | 高 |
| 末端执行器姿态 | 高 | 中 | 高 |

- LeWM **持续优于 PLDM**，接近 DINOv2 的水平
- DINOv2 在 1.24 亿张图像上预训练，而 LeWM 仅在任务数据上端到端训练

#### 4.3.2 惊讶度评估 (Surprise/Violation-of-Expectation)

| 事件类型 | 惊讶度响应 | 说明 |
|----------|-----------|------|
| **物体瞬移** (物理违反) | **显著升高** | 正确检测到不合理事件 |
| **颜色改变** (视觉扰动) | **无显著变化** | 忽略视觉表面变化 |
| 正常运动 | 基线水平 | 正常 |

这证明 LeWM 的潜在空间编码了**真正的物理理解**，而非肤浅的视觉匹配。

#### 4.3.3 t-SNE 可视化

- Push-T 的 t-SNE 可视化显示潜在空间**保持了空间邻域结构**
- 在 2D 工作空间中物理距离近的点，在潜在空间中仍然相近

---

## 五、TwoRoom 数据集复现计划

### 5.1 当前复现状态

根据 git 历史记录，当前代码库已完成：

- [x] 基础环境搭建（conda + 依赖安装）
- [x] 数据集准备（HDF5 格式）
- [x] 模型代码完整（jepa.py, module.py）
- [x] 训练脚本和配置（Hydra 配置系统）
- [x] 评估脚本（eval.py）
- [x] OOM 问题调试（batch_size 调整为 64，precision 调整）
- [x] HuggingFace 权重转换脚本（convert_pretrained.py）
- [x] **已成功训练和推理 tworoom 数据集**（commit d87c2e8）

### 5.2 TwoRoom 数据集详情

| 配置项 | 值 |
|--------|-----|
| 数据集名称 | `tworoom` → `$STABLEWM_HOME/tworoom.h5` |
| 数据键 | pixels, action, proprio, episode_idx, step_idx |
| 帧跳跃 | 5 |
| 序列长度 | history_size(3) + num_preds(1) = 4 帧 |
| 评估 episodes | 50 |
| 目标偏移步数 | 25 |
| 评估预算 | 50 步 |
| 规划 horizon | 5 |
| CEM action_block | 5 |

### 5.3 详细复现步骤

#### 阶段一：环境准备与数据下载

```bash
# 1. 创建虚拟环境
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]

# 2. 设置数据目录
export STABLEWM_HOME=~/.stable-wm/

# 3. 下载 TwoRoom 数据集（从 HuggingFace）
# 方式 A: 使用官方 HF 数据集仓库
huggingface-cli download quentinll/lewm-tworooms --local-dir $STABLEWM_HOME/hf_tworoom
tar --zstd -xvf $STABLEWM_HOME/hf_tworoom/*.tar.zst -C $STABLEWM_HOME/

# 方式 B: 使用已有数据（如果已下载）
# 确保 tworoom.h5 位于 $STABLEWM_HOME/ 下
```

#### 阶段二：从零训练 LeWM on TwoRoom

```bash
# 训练配置说明
# config/train/data/tworoom.yaml  ← 数据集配置
# config/train/model/lewm.yaml    ← 模型架构配置
# config/train/lewm.yaml          ← 主训练配置

# 修改 WandB 配置
# config/train/lewm.yaml 中 wandb.config.entity/project

# 启动训练（单卡）
python train.py data=tworoom

# 多卡训练（如果需要）
# 修改 config/train/lewm.yaml 中的 trainer.devices 和 trainer.strategy
```

**关键训练参数验证清单**:
- [ ] batch_size: 64（V100-16GB 测试通过）
- [ ] precision: 16-mixed
- [ ] history_size: 3, num_preds: 1
- [ ] SIGReg weight λ: 0.09
- [ ] lr: 5e-5, weight_decay: 1e-3
- [ ] max_epochs: 100

#### 阶段三：加载官方预训练权重评估

```bash
# 方式 A: 从 HuggingFace 下载并转换
huggingface-cli download quentinll/lewm-tworooms --local-dir $STABLEWM_HOME/hf_tworoom
python test/convert_pretrained.py  # 转换 HF 格式 → object.ckpt

# 方式 B: 从 Google Drive 下载 baseline 压缩包
# 解压后放置到 $STABLEWM_HOME/tworoom/lewm_object.ckpt

# 运行评估
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm
```

#### 阶段四：评估与分析

```bash
# 1. 使用训练好的模型评估
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm

# 2. 随机策略基线
python eval.py --config-name=tworoom.yaml policy=random

# 3. 其他 baseline 评估
# 下载 baseline 权重（PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM）
# 逐一运行评估对比
```

**评估指标记录**:
- [ ] Success Rate（50 episodes 平均）
- [ ] 规划时间
- [ ] 评估总时间
- [ ] 视频可视化（results_path 中的视频文件）

#### 阶段五：结果验证与对比

| 实验编号 | 内容 | 目的 |
|----------|------|------|
| E1 | 官方权重评估 | 确认评估流程正确 |
| E2 | 从零训练评估 | 验证训练可复现性 |
| E3 | 随机策略基线 | 下限参考 |
| E4 | PLDM baseline | 端到端方法对比 |
| E5 | DINO-WM baseline | 基础模型方法对比 |

### 5.4 预期结果与注意事项

根据论文结果，TwoRoom 是 LeWM 的**弱项**，但原因**未经实验验证**：

- LeWM 在 TwoRoom 上**低于基线方法**
- 作者在 Section 4.2 中写道：*"A possible explanation is that the low diversity and low intrinsic dimensionality of this dataset make it difficult for the encoder to match the isotropic Gaussian prior enforced by SIGReg in a high-dimensional latent space, which may lead to a less structured latent representation."*
- **重要澄清**：这只是作者的**推测性解释**，不是经过实验验证的结论。论文中的措辞全部是"possible explanation"（可能的解释）、"may lead to"（可能导致）、"potential limitation"（潜在局限）等**非确定性表述**
- 论文的消融实验（Appendix G）**全部在 PushT 上进行**，包括 SIGReg 投影数、积分节点数、嵌入维度、编码器架构等，**没有在 TwoRoom 上做任何针对性的消融或验证**
- 因此，SIGReg 高斯先验是否真的是 TwoRoom 性能差的原因，或者是否存在其他解释（如数据集覆盖不足、规划配置不适合、ViT-tiny 对低维导航的空间编码能力不足等），**目前完全未知**

> ⚠️ **复现研究价值**：正因为这个假说未被验证，在 TwoRoom 上系统性地做 λ 消融、嵌入分布分析和正则化策略对比，是**最有价值的复现研究方向**——可以首次验证或推翻作者的解释。

**复现中需关注的指标**:
- 训练损失曲线是否平滑收敛（论文 Appendix I）
- SIGReg 损失是否早期快速下降后平稳
- 成功率是否接近论文报告值

---

## 六、改进方向与对比实验方案

### 6.1 基于 TwoRoom 弱项的改进

#### 改进方向 1：自适应 SIGReg 强度

**动机**: 作者推测 SIGReg 的高斯先验可能是 TwoRoom 性能差的原因，但**从未实验验证**这一假说。这是验证或推翻该解释的最直接实验。

**方案**:
- **方案 A**: 根据环境维度自适应调整 λ：`λ_adaptive = λ_base × f(state_dim)`
- **方案 B**: 分层 SIGReg — 对不同层的嵌入使用不同的 λ
- **方案 C**: 课程学习 — 训练初期使用较大 λ 防止坍缩，后期逐渐减小释放表达力

**验证实验**（⚠️ 这些实验将**首次验证**作者的 SIGReg 假说）:
| 实验编号 | λ 设置 | 验证目的 |
|----------|--------|---------|
| EXP-1a | λ = 0.01 | 弱正则化——若性能提升，则支持"高斯先验过强"假说 |
| EXP-1b | λ = 0.05 | 中等正则化 |
| EXP-1c | λ = 0.09 (默认) | 基线（复现论文默认设置） |
| EXP-1d | λ = 0.15 | 强正则化——若性能更差，则进一步支持假说 |
| EXP-1e | 课程学习: 0.15→0.03 | 探索动态调整是否优于固定值 |
| EXP-1f | 自适应 λ = f(epoch) | 动态调整 |

**实验结果解读**:
- 若 λ 减小后 TwoRoom 性能**显著提升** → 支持"高斯先验过强"假说
- 若 λ 减小后性能**无改善甚至坍缩** → 推翻假说，需寻找其他解释
- 无论结果如何，都是有意义的发现

#### 改进方向 2：增强编码器容量

**动机**: ViT-tiny 在 TwoRoom 的低维但需要精确空间推理的任务上可能容量不足。

**方案**:
- 使用 ViT-small 替代 ViT-tiny
- 添加位置编码的改进（如 2D 位置编码）
- 尝试 ResNet-18 编码器（论文验证可行）

**验证实验**:
| 实验编号 | 编码器 | 参数量 | 预期效果 |
|----------|--------|--------|---------|
| EXP-2a | ViT-tiny (默认) | ~15M | 基线 |
| EXP-2b | ViT-small | ~40M | 更强但更慢 |
| EXP-2c | ResNet-18 | ~15M | 不同归纳偏置 |

#### 改进方向 3：改进预测器架构

**动机**: 当前 ARPredictor 使用固定 history_size=3 的位置编码，可能限制了长程预测。

**方案**:
- 增大 history_size（如 5 或 7）
- 使用相对位置编码替代绝对位置编码
- 添加时间注意力机制

**验证实验**:
| 实验编号 | history_size | 预期效果 |
|----------|-------------|---------|
| EXP-3a | 3 (默认) | 基线 |
| EXP-3b | 5 | 更多上下文 |
| EXP-3c | 7 | 最长上下文 |

#### 改进方向 4：混合正则化策略

**动机**: SIGReg 仅约束边际分布为高斯，但不约束时序一致性。

**方案**:
- 添加对比学习损失 (contrastive loss) 辅助 SIGReg
- 使用 VICReg（方差-不变-协方差正则化）替代或补充 SIGReg
- 添加时间一致性正则化

**验证实验**:
| 实验编号 | 正则化 | 预期效果 |
|----------|--------|---------|
| EXP-4a | SIGReg (默认) | 基线 |
| EXP-4b | SIGReg + VICReg | 更丰富的约束 |
| EXP-4c | SIGReg + 时间一致性 | 更平滑的轨迹 |
| EXP-4d | 仅 VICReg | 消融实验 |

### 6.2 跨环境泛化实验

#### 实验 5：跨环境迁移

**目的**: 验证 LeWM 学到的表示是否具有跨环境泛化能力。

**方案**:
1. 在 TwoRoom 上训练 → 在 PushT 上评估（微调/零样本）
2. 在 PushT 上训练 → 在 TwoRoom 上评估
3. 多环境联合训练 → 各环境评估

#### 实验 6：规划器对比

**目的**: CEM vs. 梯度优化在 TwoRoom 上的效果对比。

**方案**:
| 规划器 | Horizon | Pop Size | Iterations |
|--------|---------|----------|------------|
| CEM | 5 | 100 | 5 |
| CEM | 10 | 100 | 5 |
| Adam | 5 | N/A | 50 steps |
| Random Shooting | 5 | 500 | 1 |

### 6.3 表示质量分析实验

#### 实验 7：嵌入空间探测

**目的**: 深入分析 TwoRoom 上 LeWM 嵌入空间的特性。

**方案**:
1. 线性探测：训练线性回归从嵌入预测 agent 位置
2. 非线性探测：训练 MLP 预测物理量
3. 最近邻分析：嵌入空间的邻域结构
4. 惊讶度测试：物理违反（瞬移）vs 视觉扰动（颜色变化）

#### 实验 8：训练动态分析

**目的**: 深入理解 SIGReg 的训练动态。

**方案**:
1. 记录每 epoch 的 SIGReg 损失和预测损失
2. 分析嵌入分布的演变（均值、方差、高阶矩）
3. 可视化嵌入空间的 t-SNE 在不同 epoch 的变化
4. 分析时间直线化程度随训练的变化

### 6.4 实验优先级排序

| 优先级 | 实验 | 工作量 | 预期收益 |
|--------|------|--------|---------|
| **P0** | E1-E3: 基础评估流程验证 | 低 | 确保复现正确 |
| **P1** | EXP-1a~f: SIGReg λ 消融 | 低 | **首次验证**作者的 SIGReg 假说，最有研究价值 |
| **P1** | EXP-7: 嵌入空间探测 | 中 | 深入理解模型 |
| **P2** | EXP-3a~c: history_size 消融 | 低 | 改善预测 |
| **P2** | EXP-8: 训练动态分析 | 中 | 理解 SIGReg |
| **P3** | EXP-2a~c: 编码器对比 | 中 | 架构选择 |
| **P3** | EXP-6: 规划器对比 | 中 | 规划优化 |
| **P4** | EXP-4a~d: 混合正则化 | 高 | 方法创新 |
| **P4** | EXP-5: 跨环境迁移 | 高 | 泛化验证 |

---

## 七、总结

### 7.1 论文核心贡献

1. **极简方法**: 仅两个损失项，一个超参数，实现了稳定的端到端 JEPA 训练
2. **SIGReg**: 基于特征函数检验的高斯正则化器，提供可证明的反坍缩保证
3. **高效**: 15M 参数，单 GPU 训练，比基础模型方法快 48×
4. **物理理解**: 潜在空间编码了有意义的物理结构

### 7.2 局限性

1. **低维环境弱（未验证的假说）**: TwoRoom 上低于基线，作者推测 SIGReg 高斯先验可能是原因，但**未做任何针对性实验验证**——论文所有消融均在 PushT 上进行，没有在 TwoRoom 上测试不同 λ、不同正则化策略、或分析嵌入分布
2. **3D 复杂环境**: OGBench-Cube 略低于 DINO-WM，可能因 3D 视觉复杂性
3. **仅视觉输入**: 当前仅从像素学习，未充分利用本体感觉信息
4. **短时预测**: num_preds=1 仅预测下一步，长程预测能力未验证

### 7.3 对复现工作的建议

1. **首先确认基础流程**: 使用官方权重完成 E1 评估
2. **重点在 λ 消融**: 这是最有可能改善 TwoRoom 性能的低成本实验
3. **记录详细日志**: WandB 记录所有损失和指标，便于分析
4. **视频可视化**: 保存评估视频，直观理解模型行为
5. **系统对比**: 下载所有 baseline 权重，建立完整的性能基准

---

> **参考资源**:  
> - 论文: [arxiv.org/abs/2603.19312](https://arxiv.org/abs/2603.19312)  
> - 代码: [github.com/lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)  
> - 模型: [huggingface.co/quentinll/lewm-pusht](https://huggingface.co/quentinll/lewm-pusht)  
> - 论文摘要: [huggingface.co/papers/2603.19312](https://huggingface.co/papers/2603.19312)
