"""
训练脚本

使用Hydra配置管理训练LeWM（Learning World Model）模型。
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """
    LeWM前向传播：编码观测、预测下一状态、计算损失

    Args:
        self: 模型模块
        batch: 输入批次数据
        stage: 训练阶段（train/val）
        cfg: 配置对象

    Returns:
        包含损失和嵌入的输出字典
    """
    # 获取配置参数
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # 将NaN值替换为0（出现在序列边界）
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # 编码观测和动作
    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)，B是批次大小，T是时间步，D是嵌入维度
    act_emb = output["act_emb"]

    # 提取历史上下文
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    # 提取目标标签
    tgt_emb = emb[:, n_preds:]  # label
    # 预测未来状态
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # 计算LeWM损失
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # 记录损失
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    """
    主训练函数

    使用Hydra配置管理，设置数据集、模型和训练器，然后开始训练。

    Args:
        cfg: 配置对象
    """
    #########################
    ##       数据集配置       ##
    #########################

    # 转换配置并加载数据集
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    # 创建图像预处理器
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    # 为每列添加归一化器
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        # 计算动作编码器的输入维度
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    # 组合所有变换
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    # 随机分割数据集为训练集和验证集
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    # 创建数据加载器
    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       模型/优化器配置      ##
    ##############################

    # 实例化世界模型
    world_model = hydra.utils.instantiate(cfg.model)

    # 配置优化器和调度器
    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    # 创建数据模块
    data_module = spt.data.DataModule(train=train, val=val)
    # 创建PyTorch Lightning模块
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       训练配置       ##
    ##########################

    # 设置运行目录
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    # 配置WandB日志记录器
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    # 保存配置文件
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # 创建检查点保存回调
    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    # 创建训练器
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    # 设置检查点路径
    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    # 创建管理器并开始训练
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
