# convert_pretrained.py
"""
预训练权重转换脚本

将 HuggingFace 格式的权重转换为 eval.py 期望的 object.ckpt 格式
"""

import json
import os
import sys
from pathlib import Path

# ============ 配置宏 - 可根据需要修改 ============

# 任务名称（用于输出目录）
TASK_NAME = "tworoom"

# HuggingFace 下载目录（相对于项目根目录）
HF_DIR = "data/hf_tworoom"

# 输出目录（相对于项目根目录）
OUTPUT_DIR = "data"

# ==========================================================

import torch
import stable_worldmodel as swm


def setup_project_path():
    """将项目根目录添加到 Python 路径"""
    import sys
    from pathlib import Path

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


# 设置项目路径（在其他导入之前）
project_root = setup_project_path()


def convert_hf_to_ckpt(src_dir, task_name, output_dir=None):
    """
    转换 HuggingFace 格式的权重为 object.ckpt 格式

    Args:
        src_dir: HuggingFace下载目录（包含 config.json 和 weights.pt）
        task_name: 任务名称
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = Path(swm.data.utils.get_cache_dir())

    src = Path(src_dir)
    out_dir = Path(output_dir) / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'lewm_object.ckpt'

    # 检查源文件是否存在
    config_file = src / 'config.json'
    weights_file = src / 'weights.pt'

    if not config_file.exists():
        print(f"✗ 配置文件不存在: {config_file}")
        return False

    if not weights_file.exists():
        print(f"✗ 权重文件不存在: {weights_file}")
        return False

    # 加载配置
    with open(config_file, 'r') as f:
        cfg = json.load(f)

    print(f"任务: {task_name}")
    print(f"编码器: {cfg['encoder']['size']}")
    print(f"嵌入维度: {cfg.get('action_encoder', {}).get('emb_dim', 192)}")

    # 使用 hydra.utils.instantiate 创建模型
    print("使用 Hydra 实例化模型...")
    import hydra
    try:
        model = hydra.utils.instantiate(cfg)
    except Exception as e:
        print(f"✗ Hydra 实例化失败: {e}")
        return False

    # 加载权重
    print("加载权重...")
    weights = torch.load(weights_file, map_location='cpu', weights_only=False)
    model.load_state_dict(weights, strict=False)
    model.eval()

    # 保存为 object.ckpt 格式
    print(f"保存检查点到: {out_path}")
    torch.save(model, out_path)

    print(f"✓ 检查点已保存到: {out_path}")
    print(f"✓ 文件大小: {out_path.stat().st_size / 1024 / 1024:.2f} MB")

    return True


if __name__ == "__main__":
    # 构建 HuggingFace 目录和输出目录的绝对路径
    hf_dir_abs = project_root / HF_DIR
    output_dir_abs = project_root / OUTPUT_DIR

    print("=" * 60)
    print(f"项目根目录: {project_root}")
    print(f"HuggingFace 目录: {hf_dir_abs}")
    print(f"输出目录: {output_dir_abs}")
    print(f"任务名称: {TASK_NAME}")
    print("=" * 60)

    # 设置 STABLEWM_HOME 环境变量
    os.environ['STABLEWM_HOME'] = str(output_dir_abs)

    # 转换
    success = convert_hf_to_ckpt(hf_dir_abs, TASK_NAME, output_dir_abs)

    if success:
        print("\n" + "=" * 60)
        print("✓ 转换成功")
        print(f"\n评估时使用: policy={TASK_NAME}/lewm")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ 转换失败")
        sys.exit(1)