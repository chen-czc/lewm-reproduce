"""
批量评估脚本：遍历每个 epoch 的 checkpoint 进行推理，记录精度。

思路：复用 eval.py 和 train.py 已验证可用的代码路径：
  - 模型构建：通过 Hydra 加载 train 配置，用 hydra.utils.instantiate(cfg.model) 建模型
  - 评估逻辑：复用 eval.py 中 solver + policy + evaluate_from_dataset 的流程

用法:
    # 评估全部 epoch
    python test/eval_all_epochs.py data=tworoom

    # 只评估指定 epoch
    python test/eval_all_epochs.py data=tworoom epochs=1,5,10,20

    # 减少 episode 数量加速
    python test/eval_all_epochs.py data=tworoom num_eval=20

    # 指定输出文件
    python test/eval_all_epochs.py data=tworoom output=results.csv

    # 指定 checkpoint 目录
    python test/eval_all_epochs.py data=tworoom ckpt_dir=./checkpoints/lewm
"""

import os
import sys
import re
import csv
import time
import glob
from pathlib import Path
from functools import partial

# ─── 必须在 import mujoco 之前设置 ───
os.environ["MUJOCO_GL"] = "egl"

# 确保项目根目录在 sys.path 中（本脚本在 test/ 子目录下）
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
from omegaconf import OmegaConf, DictConfig
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import jepa        # noqa: F401 — 注册命名空间
import module      # noqa: F401


# ═══════════════════════════════════════════════════════════
#  以下函数全部从 eval.py 搬过来，不做任何修改
# ═══════════════════════════════════════════════════════════

def img_transform(img_size=224):
    """创建图像变换（eval.py:24-41）"""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def get_episodes_length(dataset, episodes):
    """获取指定 episode 的长度（eval.py:45-63）"""
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def build_normalizers(dataset, keys_to_cache):
    """构建归一化器（eval.py:117-131）"""
    process = {}
    for col in keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def sample_eval_starting_points(dataset, num_eval, goal_offset_steps, seed=42):
    """采样评估起始点（eval.py:179-208）"""
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    g = np.random.default_rng(seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    print(f"  采样到 {len(eval_episodes)} 个评估起始点")
    return eval_episodes.tolist(), eval_start_idx.tolist()


# ═══════════════════════════════════════════════════════════
#  主入口：使用 Hydra 加载 train 配置（和 train.py 完全一致）
# ═══════════════════════════════════════════════════════════

@hydra.main(version_base=None, config_path="../config/train", config_name="lewm")
def main(cfg: DictConfig):
    print("=" * 60)
    print("  LeWM 批量评估脚本（基于 eval.py + train.py 已验证流程）")
    print("=" * 60)

    # ── 解除 struct 锁定，允许命令行传入自定义参数 ──
    # Hydra 默认不允许往配置里加配置文件中不存在的 key
    # 解锁后 epochs=1, ckpt_dir=xxx 等都可以直接用，不需要 + 前缀
    OmegaConf.set_struct(cfg, False)

    # ── 从命令行或配置中读取自定义参数 ──
    ckpt_dir   = cfg.get("ckpt_dir", "./checkpoints/lewm")
    output     = cfg.get("output", "checkpoint_eval_results.csv")
    epochs_arg = cfg.get("epochs", None)   # e.g. "1,5,10,20"
    num_eval   = cfg.get("num_eval", 50)
    skip_existing = cfg.get("skip_existing", False)

    # ──────────────────────────────────────────
    #  阶段 1：扫描 checkpoint 文件
    # ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  【1/5】扫描 checkpoint 文件")
    print(f"{'─'*60}")
    print(f"  目录: {ckpt_dir}")

    pattern = os.path.join(ckpt_dir, "weights_epoch_*.pt")
    files = glob.glob(pattern)
    all_ckpts = {}
    for f in files:
        match = re.search(r"weights_epoch_(\d+)\.pt", os.path.basename(f))
        if match:
            epoch = int(match.group(1))
            all_ckpts[epoch] = f

    if not all_ckpts:
        print(f"  [!] 未找到任何 checkpoint")
        return

    all_ckpts = dict(sorted(all_ckpts.items()))
    print(f"  找到 {len(all_ckpts)} 个 checkpoint (epoch {min(all_ckpts)} ~ {max(all_ckpts)})")

    # 筛选指定 epoch
    if epochs_arg is not None:
        # Hydra 类型推断：epochs=1 → int, epochs=[1,5,10] → ListConfig, epochs="1,5" → str
        if isinstance(epochs_arg, int):
            selected_epochs = {epochs_arg}
        elif isinstance(epochs_arg, str):
            selected_epochs = set(int(e.strip()) for e in epochs_arg.split(","))
        else:
            # ListConfig 或 list
            selected_epochs = set(int(e) for e in epochs_arg)
        all_ckpts = {e: p for e, p in all_ckpts.items() if e in selected_epochs}
        print(f"  筛选后: {sorted(all_ckpts.keys())}")

    if not all_ckpts:
        print("  [!] 没有可评估的 checkpoint")
        return

    # ──────────────────────────────────────────
    #  阶段 2：加载数据集 & 构建模型
    #  （和 train.py 的流程完全一致）
    # ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  【2/5】加载数据集 & 构建模型（与 train.py 一致）")
    print(f"{'─'*60}")

    # 2a. 加载数据集（和 train.py:113-127 完全一致）
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    if cache_dir is None:
        cache_dir = swm.data.utils.get_cache_dir()
    print(f"  数据集: {dataset_name}, 缓存: {cache_dir}")
    dataset = swm.data.HDF5Dataset(
        dataset_name, cache_dir=cache_dir, **dataset_cfg
    )

    # 2b. 计算 action_encoder.input_dim（train.py:139-140）
    frameskip = cfg.data.dataset.get("frameskip", 5)
    raw_action_dim = dataset.get_dim("action")
    action_input_dim = frameskip * raw_action_dim
    OmegaConf.set_struct(cfg, False)
    cfg.model.action_encoder.input_dim = action_input_dim
    print(f"  frameskip={frameskip}, action_dim={raw_action_dim} → input_dim={action_input_dim}")

    # 2c. 构建模型（train.py:161）— Hydra 自动处理所有配置插值
    print(f"  用 hydra.utils.instantiate(cfg.model) 构建模型...")
    model = hydra.utils.instantiate(cfg.model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params / 1e6:.2f}M")
    print(f"  ✓ 模型构建完成")

    # ──────────────────────────────────────────
    #  阶段 3：初始化评估环境
    #  （和 eval.py 的流程完全一致）
    # ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  【3/5】初始化评估环境（与 eval.py 一致）")
    print(f"{'─'*60}")

    # 加载 eval 配置（手动合并 solver 子配置）
    eval_cfg_path = Path(_PROJECT_ROOT) / "config" / "eval" / "tworoom.yaml"
    eval_cfg = OmegaConf.load(eval_cfg_path)
    solver_cfg = OmegaConf.load(Path(_PROJECT_ROOT) / "config" / "eval" / "solver" / "cem.yaml")
    OmegaConf.set_struct(eval_cfg, False)
    eval_cfg.solver = solver_cfg
    # 删除 Hydra 专属字段，避免 resolve 报错
    eval_cfg.pop("defaults", None)

    # resolve eval_cfg 中的自引用（${eval.num_eval}, ${seed} 等）
    OmegaConf.resolve(eval_cfg)
    print(f"  eval 配置加载完成")

    # 3a. 创建环境（eval.py:99-101）
    eval_cfg.world.max_episode_steps = 2 * eval_cfg.eval.eval_budget
    world = swm.World(**eval_cfg.world, image_shape=(224, 224))
    print(f"  环境: {eval_cfg.world.env_name}, max_steps={eval_cfg.world.max_episode_steps}")

    # 3b. 加载评估数据集（eval.py:110）
    eval_dataset = swm.data.HDF5Dataset(
        eval_cfg.eval.dataset_name,
        keys_to_cache=eval_cfg.dataset.keys_to_cache,
        cache_dir=Path(cache_dir),
    )

    # 3c. 构建归一化器（eval.py:117-131）
    process = build_normalizers(eval_dataset, eval_cfg.dataset.keys_to_cache)
    print(f"  归一化器: {list(process.keys())}")

    # 3d. 采样评估起始点（eval.py:179-208）
    eval_episodes, eval_start_idx = sample_eval_starting_points(
        eval_dataset,
        num_eval=min(num_eval, eval_cfg.eval.num_eval),
        goal_offset_steps=eval_cfg.eval.goal_offset_steps,
        seed=eval_cfg.seed,
    )

    # 3e. callables（eval.py:228）
    callables = None
    if eval_cfg.eval.get("callables"):
        callables = OmegaConf.to_container(eval_cfg.eval.callables, resolve=True)

    # 3f. 图像变换
    transform = {
        "pixels": img_transform(eval_cfg.eval.img_size),
        "goal":   img_transform(eval_cfg.eval.img_size),
    }

    # 打印评估配置
    print(f"  评估参数: num_eval={num_eval}, budget={eval_cfg.eval.eval_budget}, "
          f"goal_offset={eval_cfg.eval.goal_offset_steps}")
    print(f"  CEM solver: samples={solver_cfg.num_samples}, iterations={solver_cfg.n_steps}, topk={solver_cfg.topk}")
    print(f"  规划参数: horizon={eval_cfg.plan_config.horizon}, action_block={eval_cfg.plan_config.action_block}")
    print(f"  ✓ 评估环境初始化完成")

    # ──────────────────────────────────────────
    #  阶段 4：断点续评检查
    # ──────────────────────────────────────────
    completed_epochs = set()
    output_path = Path(output)
    if skip_existing and output_path.exists():
        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_epochs.add(int(row["epoch"]))
        if completed_epochs:
            print(f"  跳过已评估的 epoch: {sorted(completed_epochs)}")

    # ──────────────────────────────────────────
    #  阶段 5：逐 epoch 评估
    # ──────────────────────────────────────────
    total_to_eval = len(all_ckpts) - len(completed_epochs & set(all_ckpts.keys()))
    print(f"\n{'─'*60}")
    print(f"  【5/5】逐 epoch 评估（共 {total_to_eval} 个待评估）")
    print(f"  结果输出: {output_path.resolve()}")
    print(f"{'─'*60}\n")

    results = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overall_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  计算设备: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}, "
              f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    for idx, (epoch, ckpt_path) in enumerate(all_ckpts.items()):
        if epoch in completed_epochs:
            print(f"  [{idx+1}/{len(all_ckpts)}] Epoch {epoch:3d} — ⏭  跳过（已评估）")
            continue

        epoch_start = time.time()
        print(f"  [{idx+1}/{len(all_ckpts)}] Epoch {epoch:3d} {'─'*40}")
        print(f"    权重: {ckpt_path}")

        # [a] 加载 state_dict 到模型
        print(f"    [a] 加载 state_dict...", end=" ", flush=True)
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            # 处理可能的 'model.' 前缀
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            results.append({"epoch": epoch, "success_rate": "ERROR", "eval_time": "0", "error": str(e)})
            _save_results(output_path, results)
            continue

        # [b] 设置模型为评估模式，构建 solver + policy（eval.py:147-166）
        print(f"    [b] 构建 solver + policy...", end=" ", flush=True)
        try:
            model = model.to(device).eval()
            model.requires_grad_(False)
            model.interpolate_pos_encoding = True

            plan_config = swm.PlanConfig(**eval_cfg.plan_config)
            eval_cfg.solver.device = device

            solver = hydra.utils.instantiate(eval_cfg.solver, model=model)
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=plan_config,
                process=process, transform=transform,
            )
            print(f"✓ (device={device})")
        except Exception as e:
            print(f"✗ {e}")
            results.append({"epoch": epoch, "success_rate": "ERROR", "eval_time": "0", "error": str(e)})
            _save_results(output_path, results)
            continue

        # [c] 运行评估（eval.py:218-231）
        print(f"    [c] 运行评估 ({num_eval} episodes)...", flush=True)
        world.set_policy(policy)
        start_time = time.time()
        try:
            metrics = world.evaluate_from_dataset(
                dataset=eval_dataset,
                episodes_idx=eval_episodes,
                start_steps=eval_start_idx,
                goal_offset_steps=eval_cfg.eval.goal_offset_steps,
                eval_budget=eval_cfg.eval.eval_budget,
                callables=callables,
                save_video=False,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"    ✗ 评估失败 ({elapsed:.1f}s): {e}")
            results.append({"epoch": epoch, "success_rate": "ERROR", "eval_time": "0", "error": str(e)})
            _save_results(output_path, results)
            continue

        elapsed = time.time() - start_time
        success_rate = metrics.get("success_rate", metrics)

        # 计算进度和 ETA
        done = len([r for r in results if r.get("success_rate") != "ERROR"]) + len(completed_epochs)
        overall_elapsed = time.time() - overall_start
        eta = (overall_elapsed / max(done, 1)) * (total_to_eval - done)

        print(f"    [d] success_rate = {success_rate}")
        print(f"        耗时: {elapsed:.1f}s | 进度: {done}/{total_to_eval} | "
              f"预计剩余: {eta/60:.1f}min")

        results.append({
            "epoch": epoch,
            "success_rate": success_rate,
            "eval_time": f"{elapsed:.1f}",
            "error": "",
        })
        _save_results(output_path, results)
        print(f"    [e] 已写入 {output_path.name}")

    # ──────────────────────────────────────────
    #  汇总
    # ──────────────────────────────────────────
    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  评估完成！总耗时: {total_time/60:.1f}min")
    print(f"  结果: {output_path.resolve()}")
    print(f"{'='*60}\n")
    _print_summary(results)


# ═══════════════════════════════════════════════════════════
#  结果保存 & 打印
# ═══════════════════════════════════════════════════════════

def _save_results(output_path: Path, new_results: list):
    """增量写入 CSV（每个 epoch 完成后立即保存）"""
    fieldnames = ["epoch", "success_rate", "eval_time", "error"]

    # 读取已有结果
    existing = {}
    if output_path.exists():
        with open(output_path, "r") as f:
            for row in csv.DictReader(f):
                existing[int(row["epoch"])] = row

    # 新结果覆盖旧的
    for r in new_results:
        existing[int(r["epoch"])] = {k: r.get(k, "") for k in fieldnames}

    # 按 epoch 排序写入
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in sorted(existing.keys()):
            writer.writerow(existing[epoch])


def _print_summary(results: list):
    """打印汇总表格"""
    valid = [r for r in results if r.get("success_rate") != "ERROR"]
    if not valid:
        print("  （无有效结果）")
        return

    print(f"  {'Epoch':>6s} | {'Success Rate':>12s} | {'Time (s)':>8s}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*8}")

    best_epoch, best_rate = None, -1.0
    for r in valid:
        rate = float(r["success_rate"])
        marker = ""
        if rate > best_rate:
            best_rate = rate
            best_epoch = r["epoch"]
            marker = " ← best"
        print(f"  {r['epoch']:>6d} | {rate:>12.4f} | {r['eval_time']:>8s}{marker}")

    print(f"\n  🏆 最佳 epoch: {best_epoch} (success_rate={best_rate:.4f})")
    print(f"  📊 建议: 训练时设置 max_epochs={best_epoch} 或略大即可\n")


if __name__ == "__main__":
    main()
