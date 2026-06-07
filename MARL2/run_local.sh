#!/usr/bin/env bash
# ============================================================================
# 本地一键运行脚本（Mac mini / 任意机器通用）
#
# 用法：
#     bash run_local.sh                   # 用下面配置区的默认值
#     bash run_local.sh --env-num 32      # 覆盖并行环境数
#     bash run_local.sh --env-num 16 --env-name PongNoFrameskip-v4
#
# 支持覆盖的参数（直接透传给 main.py，其余默认值见配置区）：
#     --env-num, --env-name, --gameflag, --results-dir, 以及任何 main.py 参数
#
# 它会做三件事：
#   1) git pull 拉取 GitHub 上最新的代码和实验配置
#   2) 检查依赖是否就绪（缺了会提示怎么装）
#   3) 按下面 EXPERIMENT 区块里的固定配置启动训练（命令行参数可覆盖）
# ============================================================================
set -e

# --- 切到脚本所在目录 (MARL2) ---
SCRIPT_DIR=”$(cd “$(dirname “${BASH_SOURCE[0]}”)” && pwd)”
cd “$SCRIPT_DIR”

# --- 1. 拉最新代码 ---
echo “=== 拉取最新代码 (git pull origin main) ===”
git pull origin main || echo “(git pull 失败，先用本地现有代码继续)”

# --- 2. 依赖检查 ---
echo “=== 检查依赖 ===”
if ! python -c “import stable_baselines3, ale_py, gymnasium, torch, cv2, skvideo” 2>/dev/null; then
    echo “缺少依赖。请在当前 conda 环境里执行一次：”
    echo “    pip install -r requirements.txt && pip uninstall -y gym”
    exit 1
fi

# ============================================================================
# EXPERIMENT 配置区 —— 默认值；命令行参数（见用法）可覆盖任意一项
# ============================================================================
ENV_NAME=”BreakoutNoFrameskip-v4”
GAMEFLAG=”atari”
ENV_NUM=32                     # Atari 推荐 32；本地 CPU 跑可改小（如 6）
RESULTS_DIR=”./outputs”        # 结果/曲线图输出目录（本地）
EXTRA_ARGS=”--plotscore”       # 其它开关
# ============================================================================

# --- 解析命令行覆盖参数（透传给 main.py，同时更新上面的变量用于日志打印）---
OVERRIDE_ARGS=()
while [[ $# -gt 0 ]]; do
    case “$1” in
        --env-name)   ENV_NAME=”$2”;  OVERRIDE_ARGS+=(“$1” “$2”); shift 2 ;;
        --gameflag)   GAMEFLAG=”$2”;  OVERRIDE_ARGS+=(“$1” “$2”); shift 2 ;;
        --env-num)    ENV_NUM=”$2”;   OVERRIDE_ARGS+=(“$1” “$2”); shift 2 ;;
        --results-dir) RESULTS_DIR=”$2”; OVERRIDE_ARGS+=(“$1” “$2”); shift 2 ;;
        *)            OVERRIDE_ARGS+=(“$1”); shift ;;  # 其余参数直接透传
    esac
done

echo “=== 启动训练: $ENV_NAME (env-num=$ENV_NUM) ===”
mkdir -p “$RESULTS_DIR”
python main.py \
    --env-name “$ENV_NAME” \
    --gameflag “$GAMEFLAG” \
    --env-num “$ENV_NUM” \
    --results-dir “$RESULTS_DIR” \
    $EXTRA_ARGS \
    “${OVERRIDE_ARGS[@]}”
