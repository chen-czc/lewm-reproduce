"""
批量评估脚本：遍历所有 epoch 的 checkpoint，逐一推理并记录精度

用法:
    # 评估所有 epoch（1~42）
    python eval_checkpoints.py

    # 只评估特定 epoch
    python eval_checkpoints.py --epochs 1 5 10 20

    # 指定 checkpoint 目录
    python eval_checkpoints.py --ckpt_dir ./checkpoints/lewm

    # 调整评估 episode 数量（默认 50，调小可加速）
    python eval_checkpoints.py --num_eval 20

    # 指定输出文件
    python eval_checkpoints.py --output results.csv

    # 指定 GPU
    CUDA_VISIBLE_DEVICES=0 python eval_checkpoints.py
"""

import os
import sys
import glob
import re
import csv
import time
import argparse
from pathlib import Path

# 必须在 import mujoco 相关库之前设置
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm

# 确保自定义模块可导入
sys.path.insert(0, str(Path(__file__).parent))
import jepa  # noqa: F401 — 注册到 torch 可反序列化的命名空间
import module  # noqa: F401
from module import SIGReg


# ─────────────────────── 辅助函数 ───────────────────────


def build_model_from_config(train_config_path: str, action_dim: int):
    """
    从训练配置文件实例化 JEPA 模型，并设置 action_encoder.input_dim。

    Args:
        train_config_path: 训练主配置文件路径 (config/train/lewm.yaml)
        action_dim: 动作编码器的输入维度 (= frameskip * raw_action_dim)

    Returns:
        JEPA 模型实例
    """
    cfg = OmegaConf.load(train_config_path)

    # 加载模型子配置
    model_cfg_path = Path(train_config_path).parent / "model" / "lewm.yaml"
    model_cfg = OmegaConf.load(model_cfg_path)

    # 解析 ${img_size}, ${embed_dim}, ${history_size} 等插值
    defaults = {
        "img_size": cfg.get("img_size", 224),
        "embed_dim": cfg.get("embed_dim", 192),
        "history_size": cfg.get("history_size", 3),
        "num_preds": cfg.get("num_preds", 1),
    }
    # 用简单值替换 OmegaConf 插值引用
    OmegaConf.resolve(model_cfg)
    # 手动替换未解析的变量
    model_cfg_str = OmegaConf.to_yaml(model_cfg)
    for key, val in defaults.items():
        model_cfg_str = model_cfg_str.replace(f"${{{key}}}", str(val))
    model_cfg = OmegaConf.create(model_cfg_str)

    # 设置 action_encoder 的 input_dim
    if "action_encoder" in model_cfg:
        OmegaConf.set_struct(model_cfg, False)
        model_cfg.action_encoder.input_dim = action_dim

    # 用 hydra.utils.instantiate 实例化（需要 hydra 全局配置）
    # 改为手动实例化，避免 Hydra 初始化开销
    from module import ARPredictor, Embedder, MLP

    # Encoder: ViT-tiny
    encoder = spt.backbone.utils.vit_hf(
        size=model_cfg.encoder.size,
        patch_size=model_cfg.encoder.patch_size,
        image_size=model_cfg.encoder.image_size,
        pretrained=False,
        use_mask_token=False,
    )

    # Predictor
    pred_cfg = OmegaConf.to_container(model_cfg.predictor, resolve=True)
    predictor = ARPredictor(**pred_cfg)

    # Action Encoder
    ae_cfg = OmegaConf.to_container(model_cfg.action_encoder, resolve=True)
    action_encoder = Embedder(**ae_cfg)

    # Projector
    proj_cfg = OmegaConf.to_container(model_cfg.projector, resolve=True)
    proj_cfg.pop("_target_", None)
    norm_fn_cfg = proj_cfg.pop("norm_fn", None)
    if norm_fn_cfg and isinstance(norm_fn_cfg, dict):
        norm_fn = torch.nn.BatchNorm1d
    else:
        norm_fn = torch.nn.BatchNorm1d
    projector = MLP(**proj_cfg, norm_fn=norm_fn)

    # Pred Projector
    pp_cfg = OmegaConf.to_container(model_cfg.pred_proj, resolve=True)
    pp_cfg.pop("_target_", None)
    pp_cfg.pop("norm_fn", None)
    pred_proj = MLP(**pp_cfg, norm_fn=torch.nn.BatchNorm1d)

    model = jepa.JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )

    return model


def load_weight_into_model(model, weight_path: str):
    """
    将 state_dict 加载到模型中。
    训练时保存的是 model.state_dict()（spt.Module.model 的 state_dict），
    这里需要处理可能的 key 前缀差异。
    """
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

    # 检查是否有 'model.' 前缀（spt.Module 包裹后可能产生）
    if any(k.startswith("model.") for k in state_dict.keys()):
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("model.", "", 1)] = v
        state_dict = cleaned

    model.load_state_dict(state_dict, strict=True)
    return model


def build_eval_components(cfg: DictConfig, model, process):
    """
    构建 eval.py 中的 solver + policy，复用其规划逻辑。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    config = swm.PlanConfig(**cfg.plan_config)

    # 覆盖 solver 的 device
    OmegaConf.set_struct(cfg, False)
    cfg.solver.device = device

    solver = hydra.utils.instantiate(cfg.solver, model=model)
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform,
    )
    return policy, device


def img_transform(cfg):
    """创建图像变换（与 eval.py 一致）"""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_episodes_length(dataset, episodes):
    """获取指定 episode 的长度"""
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def setup_eval_env(eval_config_path: str):
    """
    加载 eval 配置，构建评估环境、数据集和归一化器。
    返回 (cfg, world, dataset, process, eval_indices_info)
    """
    cfg = OmegaConf.load(eval_config_path)

    # 加载 solver 子配置
    solver_cfg_path = Path(eval_config_path).parent / "solver" / "cem.yaml"
    solver_cfg = OmegaConf.load(solver_cfg_path)
    OmegaConf.set_struct(cfg, False)
    cfg.solver = solver_cfg

    # 加载 launcher 子配置（可能不需要，但避免缺失报错）
    launcher_cfg_path = Path(eval_config_path).parent / "launcher" / "local.yaml"
    if launcher_cfg_path.exists():
        cfg.launcher = OmegaConf.load(launcher_cfg_path)

    # 填充必要默认值
    if "seed" not in cfg:
        cfg.seed = 42
    if "dataset" not in cfg:
        cfg.dataset = {"keys_to_cache": ["action", "proprio"], "stats": cfg.eval.dataset_name}

    # 创建环境
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # 加载数据集
    cache_dir = swm.data.utils.get_cache_dir()
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=Path(cache_dir),
    )

    # 构建归一化器
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # 采样评估起始点（与 eval.py 一致）
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    eval_info = {
        "episodes_idx": eval_episodes.tolist(),
        "start_steps": eval_start_idx.tolist(),
        "goal_offset_steps": cfg.eval.goal_offset_steps,
        "eval_budget": cfg.eval.eval_budget,
        "callables": OmegaConf.to_container(cfg.eval.get("callables"), resolve=True) if cfg.eval.get("callables") else None,
    }

    return cfg, world, dataset, process, eval_info


def find_checkpoints(ckpt_dir: str) -> dict:
    """
    扫描 checkpoint 目录，返回 {epoch: path} 的有序字典。
    """
    pattern = os.path.join(ckpt_dir, "weights_epoch_*.pt")
    files = glob.glob(pattern)
    ckpts = {}
    for f in files:
        match = re.search(r"weights_epoch_(\d+)\.pt", os.path.basename(f))
        if match:
            epoch = int(match.group(1))
            ckpts[epoch] = f

    # 按 epoch 排序
    return dict(sorted(ckpts.items()))


# ─────────────────────── 主流程 ───────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="批量评估各 epoch checkpoint 的推理精度")
    parser.add_argument(
        "--ckpt_dir", type=str, default="./checkpoints/lewm",
        help="checkpoint 所在目录 (默认: ./checkpoints/lewm)",
    )
    parser.add_argument(
        "--eval_config", type=str, default="./config/eval/tworoom.yaml",
        help="eval 配置文件路径 (默认: ./config/eval/tworoom.yaml)",
    )
    parser.add_argument(
        "--train_config", type=str, default="./config/train/lewm.yaml",
        help="train 配置文件路径，用于构建模型结构 (默认: ./config/train/lewm.yaml)",
    )
    parser.add_argument(
        "--data_config", type=str, default="./config/train/data/tworoom.yaml",
        help="数据集配置文件，用于确定 frameskip 和 action_dim (默认: ./config/train/data/tworoom.yaml)",
    )
    parser.add_argument(
        "--epochs", type=int, nargs="+", default=None,
        help="只评估指定的 epoch（例如 --epochs 1 5 10 20）。不指定则评估全部。",
    )
    parser.add_argument(
        "--num_eval", type=int, default=None,
        help="每个 epoch 评估的 episode 数量（默认使用 eval 配置中的值 50，调小可加速）",
    )
    parser.add_argument(
        "--output", type=str, default="checkpoint_eval_results.csv",
        help="结果输出 CSV 文件路径 (默认: checkpoint_eval_results.csv)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="跳过已经评估过的 epoch（读取 output 文件判断）",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="指定使用的 GPU 编号（例如 --gpu 0）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[*] 使用 GPU: {args.gpu}")

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 计算设备: {device_name}")
    if device_name == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    # 1. 扫描所有 checkpoint
    all_ckpts = find_checkpoints(args.ckpt_dir)
    if not all_ckpts:
        print(f"[!] 在 {args.ckpt_dir} 中未找到任何 checkpoint 文件")
        return

    print(f"[*] 找到 {len(all_ckpts)} 个 checkpoint (epoch {min(all_ckpts)} ~ {max(all_ckpts)})")

    # 筛选指定 epoch
    if args.epochs:
        selected = {e: p for e, p in all_ckpts.items() if e in args.epochs}
        missing = set(args.epochs) - set(selected.keys())
        if missing:
            print(f"[!] 以下 epoch 不存在: {sorted(missing)}")
        all_ckpts = selected

    if not all_ckpts:
        print("[!] 没有可评估的 checkpoint")
        return

    print(f"[*] 将评估 {len(all_ckpts)} 个 epoch: {sorted(all_ckpts.keys())}")

    # 2. 读取数据集配置，计算 action_encoder input_dim
    data_cfg = OmegaConf.load(args.data_config)
    frameskip = data_cfg.dataset.get("frameskip", 5)
    dataset_name = data_cfg.dataset.name
    print(f"[*] 数据集: {dataset_name}, frameskip: {frameskip}")

    # 加载数据集获取 action 维度
    cache_dir = swm.data.utils.get_cache_dir()
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=data_cfg.dataset.get("keys_to_cache", ["action", "proprio"]),
        cache_dir=Path(cache_dir),
    )
    raw_action_dim = dataset.get_dim("action")
    action_input_dim = frameskip * raw_action_dim
    print(f"[*] action_dim: {raw_action_dim}, frameskip: {frameskip} -> input_dim: {action_input_dim}")

    # 3. 构建模型骨架（所有 epoch 共享同一结构）
    model = build_model_from_config(args.train_config, action_input_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] 模型参数量: {total_params / 1e6:.2f}M")

    # 4. 设置评估环境（只需一次，所有 epoch 复用）
    print("[*] 初始化评估环境和数据集...")
    cfg, world, eval_dataset, process, eval_info = setup_eval_env(args.eval_config)

    # 覆盖 num_eval（如果命令行指定了更小的值，加速评估）
    if args.num_eval and args.num_eval < cfg.eval.num_eval:
        print(f"[*] 覆盖 num_eval: {cfg.eval.num_eval} -> {args.num_eval}")
        # 重新采样更少的 episode
        g = np.random.default_rng(cfg.seed)
        col_name = "episode_idx" if "episode_idx" in eval_dataset.column_names else "ep_idx"
        ep_indices_all, _ = np.unique(eval_dataset.get_col_data(col_name), return_index=True)
        episode_len = get_episodes_length(eval_dataset, ep_indices_all)
        max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
        max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices_all)}
        col_name = "episode_idx" if "episode_idx" in eval_dataset.column_names else "ep_idx"
        max_start_per_row = np.array(
            [max_start_idx_dict[ep_id] for ep_id in eval_dataset.get_col_data(col_name)]
        )
        valid_mask = eval_dataset.get_col_data("step_idx") <= max_start_per_row
        valid_indices = np.nonzero(valid_mask)[0]

        random_episode_indices = g.choice(
            len(valid_indices) - 1, size=args.num_eval, replace=False
        )
        random_episode_indices = np.sort(valid_indices[random_episode_indices])

        eval_episodes = eval_dataset.get_row_data(random_episode_indices)[col_name]
        eval_start_idx = eval_dataset.get_row_data(random_episode_indices)["step_idx"]
        eval_info["episodes_idx"] = eval_episodes.tolist()
        eval_info["start_steps"] = eval_start_idx.tolist()

    # 5. 读取已完成的 epoch（支持断点续评）
    completed_epochs = set()
    if args.skip_existing and os.path.exists(args.output):
        with open(args.output, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_epochs.add(int(row["epoch"]))
        if completed_epochs:
            print(f"[*] 跳过已评估的 epoch: {sorted(completed_epochs)}")

    # 6. 遍历评估
    results = []
    total_epochs = len(all_ckpts)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  开始批量评估 — 共 {total_epochs} 个 epoch")
    print(f"  评估配置: {args.num_eval or cfg.eval.num_eval} episodes, budget={cfg.eval.eval_budget}")
    print(f"  结果输出: {output_path}")
    print(f"{'='*60}\n")

    for idx, (epoch, ckpt_path) in enumerate(all_ckpts.items()):
        if epoch in completed_epochs:
            print(f"  [{idx+1}/{total_epochs}] Epoch {epoch:3d} — 跳过（已评估）")
            continue

        print(f"  [{idx+1}/{total_epochs}] Epoch {epoch:3d} — 加载权重: {ckpt_path}")

        # 加载权重
        try:
            load_weight_into_model(model, ckpt_path)
        except Exception as e:
            print(f"    ✗ 加载失败: {e}")
            results.append({
                "epoch": epoch, "success_rate": "ERROR", "eval_time": 0,
                "error": str(e),
            })
            continue

        # 构建 solver + policy
        try:
            policy, device = build_eval_components(cfg, model, process)
        except Exception as e:
            print(f"    ✗ 构建策略失败: {e}")
            results.append({
                "epoch": epoch, "success_rate": "ERROR", "eval_time": 0,
                "error": str(e),
            })
            continue

        # 运行评估
        world.set_policy(policy)
        start_time = time.time()
        try:
            metrics = world.evaluate_from_dataset(
                dataset=eval_dataset,
                episodes_idx=eval_info["episodes_idx"],
                start_steps=eval_info["start_steps"],
                goal_offset_steps=eval_info["goal_offset_steps"],
                eval_budget=eval_info["eval_budget"],
                callables=eval_info["callables"],
                save_video=False,  # 批量评估关闭视频，加速
            )
        except Exception as e:
            print(f"    ✗ 评估失败: {e}")
            results.append({
                "epoch": epoch, "success_rate": "ERROR", "eval_time": 0,
                "error": str(e),
            })
            continue

        elapsed = time.time() - start_time

        # 提取指标
        success_rate = metrics.get("success_rate", metrics)
        print(f"    ✓ success_rate={success_rate}, time={elapsed:.1f}s")

        result = {
            "epoch": epoch,
            "success_rate": success_rate,
            "eval_time": f"{elapsed:.1f}",
            "error": "",
        }
        results.append(result)

        # 增量写入（防止中途崩溃丢失结果）
        _write_results(output_path, results)

    # 7. 最终输出汇总
    print(f"\n{'='*60}")
    print(f"  评估完成 — 结果已保存到 {output_path}")
    print(f"{'='*60}\n")

    # 打印表格
    _print_summary(results)

    return results


def _write_results(output_path: Path, results: list):
    """将结果写入 CSV（覆盖写入，包含已完成的和新的）"""
    if not results:
        return

    fieldnames = ["epoch", "success_rate", "eval_time", "error"]
    # 读取已有结果（如果文件存在）
    existing = []
    if output_path.exists():
        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.append(row)

    # 合并：新结果覆盖旧结果
    existing_epochs = {int(r["epoch"]): r for r in existing}
    new_epochs = {int(r["epoch"]): r for r in results if isinstance(r, dict)}
    existing_epochs.update(new_epochs)

    all_results = sorted(existing_epochs.values(), key=lambda x: int(x["epoch"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _print_summary(results: list):
    """打印结果汇总表格"""
    valid = [r for r in results if isinstance(r, dict) and r.get("success_rate") != "ERROR"]
    if not valid:
        print("  （无有效结果）")
        return

    print(f"  {'Epoch':>6s} | {'Success Rate':>12s} | {'Time (s)':>8s}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*8}")

    best_epoch = None
    best_rate = -1.0

    for r in valid:
        rate = float(r["success_rate"])
        t = r["eval_time"]
        marker = ""
        if rate > best_rate:
            best_rate = rate
            best_epoch = r["epoch"]
            marker = " ← best"
        print(f"  {r['epoch']:>6d} | {rate:>12.4f} | {t:>8s}{marker}")

    print(f"\n  🏆 最佳 epoch: {best_epoch} (success_rate={best_rate:.4f})")
    print(f"  📊 建议: 训练时设置 max_epochs={best_epoch} 或略大即可\n")


if __name__ == "__main__":
    main()
