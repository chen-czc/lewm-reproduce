# validate_data.py
"""
数据集验证脚本

验证数据集文件的完整性和加载是否正常
"""

import os
import sys

# ============ 配置宏 - 可根据需要修改 ============

# 数据目录（相对于项目根目录）
DATA_DIR = "data"

# 数据集列表（不带 .h5 后缀）
DATASETS = [
    "tworoom",  # Two Rooms 任务
    # 添加其他数据集，例如：
    # "pusht_expert_train",
    # "cube_expert_train",
    # "reacher_expert_train",
]

# 需要缓存的列（仅缓存数据集中实际存在的列）
KEYS_TO_CACHE = ['action', 'proprio']  # 'state' 在 tworoom.h5 中不存在

# ==========================================================

def setup_project_path():
    """将项目根目录添加到 Python 路径"""
    import sys
    from pathlib import Path

    # 获取项目根目录（test 目录的上级目录）
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # 添加到 sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


# 设置项目路径
setup_project_path()

import stable_worldmodel as swm
import torch
import h5py


def validate_h5_file(filepath):
    """验证 HDF5 文件的完整性"""
    print(f"验证文件: {filepath}")

    if not os.path.exists(filepath):
        print(f"  ✗ 文件不存在")
        return False

    try:
        with h5py.File(filepath, 'r') as f:
            print(f"  ✓ 文件可打开")
            keys = list(f.keys())
            print(f"  数据集列: {keys}")

            # 显示每个列的形状和类型
            for k in keys:
                shape = f[k].shape
                dtype = f[k].dtype
                print(f"    {k}: shape={shape}, dtype={dtype}")

            # 样本数
            first_key = keys[0]
            sample_count = len(f[first_key])
            print(f"  样本数: {sample_count}")
        return True
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return False


def validate_dataset(dataset_name, cache_dir):
    """验证数据集加载"""
    print(f"\n验证数据集: {dataset_name}")

    try:
        dataset = swm.data.HDF5Dataset(
            dataset_name,
            keys_to_cache=KEYS_TO_CACHE,
            cache_dir=cache_dir,
        )

        print(f"  ✓ 数据集加载成功")
        print(f"  列名: {dataset.column_names}")
        print(f"  长度: {len(dataset)}")

        # 采样一条数据
        sample = dataset[0]
        print(f"  采样一条数据:")
        for k, v in sample.items():
            if hasattr(v, 'shape'):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")

        return True
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 获取项目根目录（本文件的上级目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 数据目录绝对路径
    data_dir_abs = os.path.join(project_root, DATA_DIR)

    print(f"项目根目录: {project_root}")
    print(f"数据目录: {data_dir_abs}")
    print(f"待验证数据集: {DATASETS}")
    print("=" * 60)

    # 验证所有数据集
    all_passed = True
    for dataset_name in DATASETS:
        h5_filename = f"{dataset_name}.h5"
        h5_filepath = os.path.join(data_dir_abs, h5_filename)

        # 验证 HDF5 文件
        if not validate_h5_file(h5_filepath):
            all_passed = False
            continue

        # 验证数据集加载
        if not validate_dataset(dataset_name, data_dir_abs):
            all_passed = False

    if all_passed:
        print("\n" + "=" * 60)
        print("✓ 所有验证通过")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ 部分验证失败")
        sys.exit(1)