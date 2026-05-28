"""
评估脚本

用于评估世界模型策略与随机策略的性能。
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def img_transform(cfg):
    """
    创建图像变换

    Args:
        cfg: 配置对象，包含eval.img_size

    Returns:
        图像变换组合
    """
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    """
    获取指定episode的长度

    Args:
        dataset: 数据集对象
        episodes: episode ID列表

    Returns:
        每个episode长度的数组
    """
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    """
    加载数据集

    Args:
        cfg: 配置对象
        dataset_name: 数据集名称

    Returns:
        HDF5数据集对象
    """
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


@hydra.main(version_base=None, config_path="./config/eval", config_name="tworoom")
def run(cfg: DictConfig):
    """
    运行评估，比较世界模型策略与随机策略

    Args:
        cfg: 配置对象
    """
    # 验证规划配置
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # 创建世界环境
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # 创建变换
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    # 加载数据集
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    # 获取所有唯一的episode索引
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    # 为每列创建归一化处理器
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        # 移除包含NaN的行
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        # 为非动作列也创建目标处理器
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- 运行评估
    policy = cfg.get("policy", "random")

    # 如果不是随机策略，加载预训练模型
    if policy != "random":
        # 修复：绕过官方缺失的 utils 模块，直接手动拼接路径加载
        model_path = Path(swm.data.utils.get_cache_dir()) / f"{cfg.policy}_object.ckpt"
        print(f"[*] 正在加载本地模型: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"找不到权重文件，请检查路径: {model_path}")
            
        # 修复 PyTorch 2.6+ 的安全限制，允许加载完整的模型对象
        import jepa  # 确保命名空间里有这个类
        import module
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        
        # 自动检测，如果没有可用/兼容的 GPU 就退回到 CPU
        device = "cpu"
        model = model.to(device)
        print(f"[*] 模型已加载至: {device}")

        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        
        # 强制将 solver 的配置改为当前设备 (CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        OmegaConf.set_struct(cfg, False) # 解除 Hydra 配置锁定
        cfg.solver.device = device       # 强制覆盖 solver 的 device
        
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    # 设置结果保存路径
    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # 采样episode和起始索引
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    # 为每个episode创建最大起始索引映射
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # 将每个数据行的episode_idx映射到其max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # 移除所有step_idx > max_start_per_row的数据行
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    # 随机采样评估起始点
    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # 递增排序以避免HDF5Dataset索引问题
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    # 获取评估episode和起始索引
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    # 设置策略
    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    # 运行评估
    start_time = time.time()
    
    # 方案A：直接调用专门处理离线数据集的评估方法
    print("[*] 正在从数据集中加载初始状态和目标状态...")
    metrics = world.evaluate_from_dataset(
        dataset=dataset,  # 直接传入原生的 dataset，不需要 NumpyDatasetWrapper
        episodes_idx=eval_episodes.tolist(),
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True) if cfg.eval.get("callables") else None,
        save_video=True,
        video_path=results_path
    )
        
    end_time = time.time()

    print(metrics)

    # 保存评估结果
    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # 与之前的运行分隔

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()