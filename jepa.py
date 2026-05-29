"""
JEPA (Joint-Embedding Predictive Architecture) 实现

包含编码器、预测器和动作编码器，用于预测未来的状态嵌入。
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    """
    分离并克隆张量，防止梯度传播

    Args:
        v: 输入张量或值

    Returns:
        分离并克隆后的张量，或原始值（如果不是张量）
    """
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):
    """
    联合嵌入预测架构 (JEPA)

    包含观测编码器、预测器和动作编码器，用于学习状态表示和预测。

    Attributes:
        encoder: 观测编码器（通常是视觉编码器）
        predictor: 预测器网络
        action_encoder: 动作编码器
        projector: 嵌入投影器
        pred_proj: 预测投影器
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        """
        初始化JEPA模型

        Args:
            encoder: 观测编码器
            predictor: 预测器
            action_encoder: 动作编码器
            projector: 可选的嵌入投影器
            pred_proj: 可选的预测投影器
        """
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """
        将观测和动作编码为嵌入

        Args:
            info: 包含'pixels'和'action'键的字典

        Returns:
            更新后的info字典，包含'emb'和'act_emb'
        """
        pixels = info['pixels'].float()
        b = pixels.size(0)
        # 展平批次和时间维度以进行编码
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        # 获取CLS token作为表示
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        # 编码动作
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """
        预测下一个状态嵌入

        Args:
            emb: 当前状态嵌入，形状为 (B, T, D)
            act_emb: 动作嵌入，形状为 (B, T, A_emb)

        Returns:
            预测的状态嵌入
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## 推理专用方法 ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """
        根据初始信息和动作序列进行模型推理

        Args:
            info: 初始信息字典，包含'pixels'键
            action_sequence: 动作序列，形状为 (B, S, T, action_dim)
                - B: 批次大小
                - S: 动作规划候选样本数
                - T: 时间步数
            history_size: 用于预测的历史长度

        Returns:
            更新后的info字典，包含'predicted_emb'
        """
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        # 分离历史动作和未来动作
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # 复制并编码初始信息
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        # 扩展到所有样本维度
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # 展平批次和样本维度以便推理
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # 自回归预测n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # 预测最后一个状态
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # 恢复批次和样本维度
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """
        计算预测嵌入和目标嵌入之间的代价

        Args:
            info_dict: 包含'predicted_emb'和'goal_emb'的字典

        Returns:
            每个动作候选的代价，形状为 (B, S)
        """
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # 返回每个动作候选的最后一步代价
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """
        给定目标和初始状态，计算动作候选的代价

        Args:
            info_dict: 包含目标信息的字典
            action_candidates: 动作候选序列

        Returns:
            每个动作候选的代价
        """
        assert "goal" in info_dict, "goal not in info_dict"

        # 将所有张量移动到模型所在设备
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        # 准备目标编码
        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        # 处理目标前缀的键
        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        # 获取目标嵌入并进行推理
        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost
