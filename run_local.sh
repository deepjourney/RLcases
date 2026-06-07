#!/usr/bin/env bash
# ============================================================================
# 本地一键运行脚本（Mac mini / 任意机器通用）
#
# 用法：
#     bash run_local.sh
#
# 它会做三件事：
#   1) git pull 拉取 GitHub 上最新的代码和实验配置
#   2) 检查依赖是否就绪（缺了会提示怎么装）
#   3) 按下面 EXPERIMENT 区块里的固定配置启动训练
#
# 以后要换实验 / 调参数，都在“EXPERIMENT 配置区”里改，提交到 GitHub，
# 你本地只需要重新跑 `bash run_local.sh` 即可。
# ============================================================================
set -e

# --- 切到脚本所在目录下的 MARL2 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 1. 拉最新代码 ---
echo "=== 拉取最新代码 (git pull origin main) ==="
git pull origin main || echo "(git pull 失败，先用本地现有代码继续)"
cd "$SCRIPT_DIR/MARL2"

# --- 2. 依赖检查 ---
echo "=== 检查依赖 ==="
if ! python -c "import stable_baselines3, ale_py, gymnasium, torch, cv2, skvideo" 2>/dev/null; then
    echo "缺少依赖。请在当前 conda 环境里执行一次："
    echo "    pip install -r requirements.txt && pip uninstall -y gym"
    exit 1
fi

# ============================================================================
# EXPERIMENT 配置区 —— 要换实验/调参就改这里（由 Claude 在 GitHub 上维护）
# ============================================================================
ENV_NAME="BreakoutNoFrameskip-v4"
GAMEFLAG="atari"
ENV_NUM=6                      # M2 Pro 性能核数；并行环境数
RESULTS_DIR="./outputs"        # 结果/曲线图输出目录（本地）
EXTRA_ARGS="--plotscore"       # 其它开关
# ============================================================================

echo "=== 启动训练: $ENV_NAME (env-num=$ENV_NUM) ==="
mkdir -p "$RESULTS_DIR"
python main.py \
    --env-name "$ENV_NAME" \
    --gameflag "$GAMEFLAG" \
    --env-num "$ENV_NUM" \
    --results-dir "$RESULTS_DIR" \
    $EXTRA_ARGS
