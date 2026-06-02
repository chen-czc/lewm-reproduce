#!/bin/bash
# SBATCH --job-name=train_tworoom    # 为你的作业起个名字，方便在队列里查找
# SBATCH --partition=v100x           # 指定你要使用的计算队列
# SBATCH --nodes=2                   # 申请 2 个计算节点
# SBATCH --ntasks-per-node=4         # 运行 4 个任务
# SBATCH --gres=gpu:4                # 申请 4 张 GPU
# SBATCH --cpus-per-task=8           # 为数据加载申请 8 个 CPU 核心（防止 CPU 成为显卡训练的瓶颈）
# SBATCH --output=slurm_log/slurm-%j.out       # 标准输出日志文件，%j 会自动替换为真实的作业 ID
# SBATCH --error=slurm-%j.err        # 错误报错日志文件（强推分离报错和常规输出，方便排查排错）


# 1. 激活你刚才配置好的完美虚拟环境 (后台作业不会自动加载你的 bashrc，必须手动 source)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lewm

export STABLEWM_HOME=/HOME/sysu_ntan/sysu_ntan_5/chenzc/le-wm/data

# 2. 进入你的代码目录
cd ~/chenzc/le-wm

# 3. 打印一下当前环境，确认使用的是刚才配好的 12.4 驱动下的 PyTorch
echo "=== 环境检查 ==="
python -c "import torch; print('PyTorch Version:', torch.__version__)"
echo "================"

# 4. 运行全量训练命令 (这里假设你的训练入口是 train.py，请根据实际情况调整)
yhrun --nodes=2 --ntasks-per-node=4 python train.py data=tworoom 