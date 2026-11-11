# Psi0＋SONIC 标称任务核查

2026-11-11：按项目需求，先验证现成权重推理，不运行训练或微调。

## 冻结候选

- SIMPLE 环境：`simple/G1WholebodyXMoveBendCarryBoxSonic-v0`。
- [Psi0 源码](https://github.com/physical-superintelligence-lab/Psi0)：`805fcda9c9654e272be2a6673fe5290944fcfb04`。
- [任务权重](https://huggingface.co/USC-PSI-Lab/psi-model/tree/main/psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223)：仓库版本 `74d7abaa61b51dfe4e95f539a83738c276ce27b2`，`ckpt_40000/model.safetensors`（6,254,227,440 字节）。
- [原始评测数据](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval)：版本 `c8258285d134d0da644fdbf1a304988398fc3819`，`G1WholebodyXMoveBendCarryBoxSonic-v0.zip`。必须保留 episodes 的 environment_config，不能拿训练转换数据代替。
- SONIC 源码与 gear_sonic Python 接口采用 SIMPLE 子模块固定提交。

任务 checkpoint 包含 VLM 与动作头；推理加载器按严格匹配载入两部分，只另需 Qwen3-VL-2B 的配置和 processor。动作维度 78（64 body token＋14 手关节），状态 43，预测 horizon 30。加载成功仍不等于任务成功。

## 隔离运行

沿用现有项目根目录。显式设置以下变量；不要执行会写入 HOME/tools 或更新原有虚拟环境的完整 teleop 安装脚本。

```bash
export SONIC_TOOLS_DIR="$PWD/.tools/sonic"
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache/runtime"
export EVAL_WBC_LOG_DIR="$PWD/experiments/reproduction/psi0-sonic/logs"
export BUILD_JOBS=4
# CUDAToolkit_ROOT 指向项目内、与 TensorRT 匹配的 CUDA；不要覆盖系统 CUDA。
bash scripts/build_sonic_controller.sh
```

Psi0 使用独立 Python 3.11 环境；gear_sonic 使用 Python 3.10。评测入口支持 `EVAL_WBC_PYTHON`，控制器环境支持 `SONIC_ENV_FILE`。不向系统 bin 安装 just 或伪装 clang。

## 验收顺序与状态

1. 核查公开权重和元数据：已找到任务专用 checkpoint，代码支持严格加载；尚待实际加载。
2. SONIC 控制器构建、资产与回放：待验证。
3. Psi0 权重严格加载、健康检查、真实观测单次推理：待验证。
4. 标称任务闭环：固定 DR level 0、原始实例，保存所有尝试的 reward、终止原因、视频、配置与版本。

先以纯 MuJoCo 检查完整链路。该视觉域与官方 Isaac 渲染不同，不能把其成功率称为官方基线复现；Isaac 版仍需 RTX 条件。回放成功不能替代 Psi0 闭环成功，单个闭环成功不能证明稳定性。

运行记录将在实际测试后补充；当前没有已通过标称任务的结论。

## 当前安装记录

运行目录由项目路径派生：`.venv-psi0`（Python 3.11）、`.venv-sonic`（Python 3.10 的增量环境）、`.tools/sonic`（GCC 13 / CMake 3.31 / TensorRT 10.9 CUDA 11）、`.tools/psi0-source/Psi0`（冻结源码）、`data/checkpoints/psi0-sonic-box`（原始权重与配置）、`data/simple-eval`（原始实例）。不运行 `uv sync` 覆盖原有 `.venv`。

Psi0 的推理依赖见 `psi0-inference-requirements.txt`；用 `uv --no-config pip install` 忽略 SIMPLE 默认 CUDA 12 索引，明确固定 `torch==2.7.0+cu118`。下载失败时可在本地准备 wheel 后传到 `.cache/psi0-wheelhouse`，用 `--no-index --find-links` 安装。编译支持 `TRT_CUDA_MAJOR=11` 和 `SONIC_DOWNLOADS_DIR`。

评测前 `source scripts/reproduction/sonic_env.sh` 设置本进程的项目内缓存、动态库、Python 入口。Psi0 服务使用独立进程；评测与控制器日志写入 `experiments/reproduction/psi0-sonic`。

预定命令（实际结果以验证记录为准）：

```bash
source scripts/reproduction/sonic_env.sh
export PYTHONPATH="$PWD/.tools/psi0-source/Psi0/src"
.venv-psi0/bin/python -m psi.deploy.serve_psi0_sonic_http \
  --policy psi0 --host 127.0.0.1 --port 8014 --ckpt-step 40000 \
  --run-dir "$PWD/data/checkpoints/psi0-sonic-box" \
  --rtc-mode test_time --action-exec-horizon 24
```

另一个任务 shell：

```bash
source scripts/reproduction/sonic_env.sh
bash scripts/run_eval_wbc.sh simple/G1WholebodyXMoveBendCarryBoxSonic-v0 replay \
  --data-dir data/simple-eval/G1WholebodyXMoveBendCarryBoxSonic-v0/dr-level-0 \
  --episode-start 0 --num-episodes 1 --dr-level 0 --sim-mode mujoco \
  --eval-dir experiments/reproduction/psi0-sonic/replay-0
# 接着独立启动控制器，以 psi0 替换 replay，使用新的 eval-dir。
```

权重 SHA256：`b60c11f84887fc8b6307851c23dc9169cef4c7e292fcaaed706d6254952bc711`（本地和服务器均已核实）。Qwen 配置/processor 缓存版本：`89644892e4d85e24eaac8bacfd4f463576704203`。SONIC 模型仓库版本：`6733128a3d8a523b1418b06bca3cdf61c8b0987f`；G1 资源仓库版本：`47667c1860fbd55d98d74a0d23d1d5d773a6e9ee`。
