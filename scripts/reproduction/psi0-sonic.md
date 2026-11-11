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
