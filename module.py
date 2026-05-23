"""
神经网络模块定义

包含各种Transformer组件、正则化器和预测器网络。
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x, shift, scale):
    """
    AdaLN-zero调制

    用于自适应层归一化，通过缩放和平移对输入进行调制。

    Args:
        x: 输入张量
        shift: 平移参数
        scale: 缩放参数

    Returns:
        调制后的张量
    """
    return x * (1 + scale) + shift


class SIGReg(torch.nn.Module):
    """
    草图各向同性高斯正则化器（单GPU）

    使用Epps-Pulley统计量来惩罚嵌入的各向异性。

    Attributes:
        num_proj: 随机投影的数量
        t: 用于计算统计量的时间点向量
        phi: 目标高斯窗口函数
        weights: 积分权重
    """

    def __init__(self, knots=17, num_proj=1024):
        """
        初始化正则化器

        Args:
            knots: 用于积分的节点数
            num_proj: 随机投影的数量
        """
        super().__init__()
        self.num_proj = num_proj
        # 创建时间点向量（0到3之间均匀分布）
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        # 创建积分权重（梯形法则）
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        # 高斯窗口函数
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        前向传播，计算Epps-Pulley统计量

        Args:
            proj: 投影后的嵌入，形状为 (T, B, D)，其中T是时间步，B是批次大小，D是维度

        Returns:
            正则化损失值
        """
        # 采样随机投影矩阵
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # 计算Epps-Pulley统计量
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # 对投影和时间步取平均


class FeedForward(nn.Module):
    """
    Transformer中使用的前馈网络

    由两个线性层组成，中间使用GELU激活函数和Dropout。

    Attributes:
        net: 前馈网络序列
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        """
        初始化前馈网络

        Args:
            dim: 输入和输出维度
            hidden_dim: 隐藏层维度
            dropout: Dropout比例
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            输出张量
        """
        return self.net(x)

class Attention(nn.Module):
    """
    带因果掩码的缩放点积注意力机制

    实现标准的自注意力机制，支持因果掩码以防止信息泄露。

    Attributes:
        heads: 注意力头的数量
        scale: 缩放因子，为 1 / sqrt(dim_head)
        dropout: Dropout比例
        norm: 层归一化
        attend: Softmax函数
        to_qkv: 将输入转换为Q、K、V的线性层
        to_out: 输出投影层
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        """
        初始化注意力层

        Args:
            dim: 输入维度
            heads: 注意力头数量
            dim_head: 每个注意力头的维度
            dropout: Dropout比例
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (B, T, D)，其中B是批次大小，T是序列长度，D是维度
            causal: 是否使用因果掩码

        Returns:
            输出张量
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        # 计算Q、K、V
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        # 执行缩放点积注意力
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """
    带AdaLN-zero条件化的Transformer块

    使用条件信息来调制层归一化的参数。

    Attributes:
        attn: 注意力层
        mlp: 前馈网络
        norm1: 第一个层归一化（不带仿变换）
        norm2: 第二个层归一化（不带仿变换）
        adaLN_modulation: 自适应层归一化调制网络
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        """
        初始化条件Transformer块

        Args:
            dim: 输入维度
            heads: 注意力头数量
            dim_head: 每个注意力头的维度
            mlp_dim: MLP隐藏层维度
            dropout: Dropout比例
        """
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # AdaLN调制网络：输出6个参数（shift/scale/gate用于注意力和MLP各一组）
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        # 初始化为零，保证训练开始时模块是恒等映射
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        """
        前向传播

        Args:
            x: 输入张量
            c: 条件信息

        Returns:
            输出张量
        """
        # 从条件信息中生成调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # 使用AdaLN调制，然后通过残差连接
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """
    标准Transformer块

    包含自注意力和前馈网络，使用残差连接和层归一化。

    Attributes:
        attn: 注意力层
        mlp: 前馈网络
        norm1: 第一个层归一化
        norm2: 第二个层归一化
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        """
        初始化标准Transformer块

        Args:
            dim: 输入维度
            heads: 注意力头数量
            dim_head: 每个注意力头的维度
            mlp_dim: MLP隐藏层维度
            dropout: Dropout比例
        """
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            输出张量
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """
    支持AdaLN-zero块的标准Transformer

    可以使用标准Block或条件ConditionalBlock构建Transformer。

    Attributes:
        norm: 最终层归一化
        layers: Transformer块列表
        input_proj: 输入投影层
        cond_proj: 条件投影层
        output_proj: 输出投影层
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        """
        初始化Transformer

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            depth: Transformer块的数量
            heads: 注意力头数量
            dim_head: 每个注意力头的维度
            mlp_dim: MLP隐藏层维度
            dropout: Dropout比例
            block_class: 使用的块类型（Block或ConditionalBlock）
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        # 输入投影（维度不匹配时使用）
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        # 条件投影（维度不匹配时使用）
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        # 输出投影（维度不匹配时使用）
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        # 堆叠Transformer块
        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        """
        前向传播

        Args:
            x: 输入张量
            c: 条件信息（ConditionalBlock需要）

        Returns:
            输出张量
        """
        # 输入投影
        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        # 条件投影
        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        # 逐层处理
        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        # 输出投影
        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    """
    嵌入层，将输入特征转换为嵌入向量

    使用1D卷积和MLP进行特征嵌入。

    Attributes:
        patch_embed: 1D卷积层
        embed: 嵌入MLP
    """

    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        """
        初始化嵌入层

        Args:
            input_dim: 输入维度
            smoothed_dim: 平滑后的维度
            emb_dim: 嵌入维度
            mlp_scale: MLP隐藏层扩展倍数
        """
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (B, T, D)，其中B是批次大小，T是时间步，D是维度

        Returns:
            嵌入张量
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """
    简单的多层感知机，支持可选的归一化和激活函数

    Attributes:
        net: 神经网络序列
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        """
        初始化MLP

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度，默认与输入相同
            norm_fn: 归一化函数，默认为LayerNorm
            act_fn: 激活函数，默认为GELU
        """
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (B*T, D)

        Returns:
            输出张量
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """
    自回归预测器，用于预测下一步的嵌入

    使用带位置编码和条件化的Transformer进行预测。

    Attributes:
        pos_embedding: 可学习位置嵌入
        dropout: Dropout层
        transformer: 条件化Transformer
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        """
        初始化自回归预测器

        Args:
            num_frames: 最大帧数（位置嵌入长度）
            depth: Transformer块的数量
            heads: 注意力头数量
            mlp_dim: MLP隐藏层维度
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度，默认与输入相同
            dim_head: 每个注意力头的维度
            dropout: Transformer内部Dropout比例
            emb_dropout: 嵌入Dropout比例
        """
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (B, T, d)，其中B是批次大小，T是时间步，d是维度
            c: 条件信息（动作），形状为 (B, T, act_dim)

        Returns:
            预测的输出张量
        """
        T = x.size(1)
        # 添加位置编码
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        # 通过条件化Transformer
        x = self.transformer(x, c)
        return x
