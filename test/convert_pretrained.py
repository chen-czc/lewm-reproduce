# convert_pretrained.py
"""
预训练权重转换脚本

将 HuggingFace 格式的权重转换为 eval.py 期望的 object.ckpt 格式
"""

import json
import os
import sys
import inspect
from pathlib import Path

# ================= 核心修改 =================
# 在导入任何自定义模块之前，获取当前脚本的上一级目录，并将其加入系统路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
# ============================================

# ============ 配置宏 - 可根据需要修改 ============

# 任务名称（用于输出目录）
TASK_NAME = "tworoom"

# HuggingFace 下载目录（相对于项目根目录）
HF_DIR = "data/hf_tworoom"

# 输出目录（相对于项目根目录，最终会放在 $STABLEWM_HOME/ 下）
OUTPUT_DIR = "checkpoint"

# ==========================================================

import torch
import stable_worldmodel as swm


def init_from_config(cls, cfg_dict):
    """
    智能实例化函数：根据目标类的签名，自动过滤掉不需要的 kwargs。
    这样可以完美避开 config.json 中的 '_target_' 等 Hydra 专用元数据。
    """
    sig = inspect.signature(cls)
    valid_params = set(sig.parameters.keys())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    
    filtered_kwargs = {}
    for k, v in cfg_dict.items():
        if k.startswith('_'):
            continue
        if k in valid_params or has_varkw:
            filtered_kwargs[k] = v
            
    return cls(**filtered_kwargs)


def convert_hf_to_ckpt(src_dir, task_name, output_dir=None):
    """
    转换 HuggingFace 格式的权重为 object.ckpt 格式
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

    # 导入模型结构
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    import stable_pretraining as spt

    print("创建模型结构...")
    encoder = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"],
        patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"],
        pretrained=False,
        use_mask_token=False,
    )

    mlp_factory = lambda k: MLP(
        input_dim=cfg[k]["input_dim"],
        output_dim=cfg[k]["output_dim"],
        hidden_dim=cfg[k]["hidden_dim"],
        norm_fn=torch.nn.BatchNorm1d
    )

    # 实例化网络组件
    model = JEPA(
        encoder=encoder,
        predictor=init_from_config(ARPredictor, cfg["predictor"]),
        action_encoder=init_from_config(Embedder, cfg.get("action_encoder", {})),
        projector=mlp_factory("projector"),
        pred_proj=mlp_factory("pred_proj"),
    )

    # 加载权重
    print("加载权重...")
    raw_weights = torch.load(weights_file, map_location='cpu', weights_only=False)
    
    # 1. 智能解包
    if 'model_state_dict' in raw_weights:
        state_dict = raw_weights['model_state_dict']
    elif 'state_dict' in raw_weights:
        state_dict = raw_weights['state_dict']
    elif 'model' in raw_weights:
        state_dict = raw_weights['model']
    else:
        state_dict = raw_weights

    # 2. 移除前缀并进行名称映射（解决 HF ViT 到 Custom ViT 的对齐问题）
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('_orig_mod.', '')
        clean_k = clean_k.replace('module.', '')
        
        # ---------- 关键修复：HuggingFace ViT 字段映射 ----------
        # a. 修正网络层前缀
        clean_k = clean_k.replace('encoder.encoder.layer.', 'encoder.layers.')
        
        # b. 修正 Attention 子模块
        clean_k = clean_k.replace('.attention.attention.query.', '.attention.q_proj.')
        clean_k = clean_k.replace('.attention.attention.key.', '.attention.k_proj.')
        clean_k = clean_k.replace('.attention.attention.value.', '.attention.v_proj.')
        clean_k = clean_k.replace('.attention.output.dense.', '.attention.o_proj.')
        
        # c. 修正 MLP (FeedForward) 子模块
        clean_k = clean_k.replace('.intermediate.dense.', '.mlp.fc1.')
        clean_k = clean_k.replace('.output.dense.', '.mlp.fc2.')
        # -------------------------------------------------------
        
        clean_state_dict[clean_k] = v

    # 3. 严格加载权重
    try:
        model.load_state_dict(clean_state_dict, strict=True)
    except RuntimeError as e:
        print("✗ 权重对齐失败！部分报错信息如下：")
        print(e)
        return False

    model.eval()

    # 保存为 object.ckpt 格式
    print(f"保存检查点到: {out_path}")
    torch.save(model, out_path)

    print(f"✓ 检查点已保存到: {out_path}")
    print(f"✓ 文件大小: {out_path.stat().st_size / 1024 / 1024:.2f} MB")

    return True


if __name__ == "__main__":
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

    # 执行转换
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