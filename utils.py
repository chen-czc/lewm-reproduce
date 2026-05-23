"""
工具函数模块

包含数据预处理、归一化和检查点保存等功能。
"""

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    """
    获取图像预处理器

    Args:
        source: 输入数据源名称
        target: 输出数据目标名称
        img_size: 图像调整大小后的尺寸，默认为224

    Returns:
        组合的图像变换对象
    """
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)



class ZScoreNormalizer:
    """
    Z分数归一化器（可序列化）

    使用类而不是闭包，以便在DataLoader工作进程生成时能够通过pickle序列化
    （LanceDataset要求）。

    Attributes:
        mean: 均值
        std: 标准差
    """

    def __init__(self, mean, std):
        """
        初始化归一化器

        Args:
            mean: 归一化使用的均值
            std: 归一化使用的标准差
        """
        self.mean = mean
        self.std = std

    def __call__(self, x):
        """
        执行Z分数归一化

        Args:
            x: 输入张量

        Returns:
            归一化后的张量，公式为 (x - mean) / std
        """
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """
    获取数据集中特定列的归一化器

    Args:
        dataset: 数据集对象
        source: 输入列名
        target: 输出列名

    Returns:
        包装后的Z分数归一化变换对象
    """
    # 获取列数据
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    # 移除包含NaN值的行
    data = data[~torch.isnan(data).any(dim=1)]
    # 计算均值和标准差
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)


class SaveCkptCallback(Callback):
    """
    检查点保存回调

    在每个epoch结束后使用save_pretrained保存模型检查点。

    Attributes:
        run_name: 运行名称
        cfg: 模型配置
        epoch_interval: 保存检查点的epoch间隔，默认为1
    """

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        """
        初始化回调

        Args:
            run_name: 运行名称
            cfg: 模型配置字典
            epoch_interval: 每隔多少个epoch保存一次检查点
        """
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        """
        训练epoch结束时调用

        Args:
            trainer: PyTorch Lightning训练器
            pl_module: PyTorch Lightning模块
        """
        super().on_train_epoch_end(trainer, pl_module)

        # 仅在主进程上执行
        if trainer.is_global_zero:
            # 每隔epoch_interval个epoch保存一次
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # 最后一个epoch也要保存
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        """
        保存模型检查点

        Args:
            model: 要保存的模型
            epoch: 当前epoch编号
        """
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )