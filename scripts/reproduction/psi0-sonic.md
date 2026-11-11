# Psi0＋SONIC 标称任务核查

2026-11-11。本阶段使用公开、已针对任务训练完成的权重，只做部署和推理；没有运行训练、微调或更新模型参数。目标是核实已有模型能否成为 SIMPLE-Phys 的标称基线。

## 结论边界

- 公开任务 checkpoint、配置和归一化统计齐全；VLM 和动作头已严格加载成功，真实 MuJoCo 观测推理通过。
- SONIC C++ 控制器已构建并实际运行，6 项 lockstep 测试通过。
- 单次推理和控制通信通过，不等于标称任务成功。完整任务要求抓起箱子、搬运到桌面并松手放置，只有抬升不能计为成功。
- 当前使用纯 MuJoCo 渲染；不能把其成功率称为官方 Isaac 视觉基线的复现。未开展物理随机化，也不因当前失败而转向训练。

## 冻结来源

| 组件 | 来源和固定版本 |
| --- | --- |
| 环境 | `simple/G1WholebodyXMoveBendCarryBoxSonic-v0`，DR level 0 |
| Psi0 源码 | [physical-superintelligence-lab/Psi0](https://github.com/physical-superintelligence-lab/Psi0)，`805fcda9c9654e272be2a6673fe5290944fcfb04` |
| 任务权重 | [USC-PSI-Lab/psi-model](https://huggingface.co/USC-PSI-Lab/psi-model/tree/74d7abaa61b51dfe4e95f539a83738c276ce27b2/psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223)，`ckpt_40000/model.safetensors` |
| 原始评测实例 | [USC-PSI-Lab/psi-data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/c8258285d134d0da644fdbf1a304988398fc3819/simple-eval)，`G1WholebodyXMoveBendCarryBoxSonic-v0.zip`，5 个实例 |
| SONIC C++ | SIMPLE 固定子模块 `948022a6fd19bd4ad10186d587b632482453b9a2` |
| gear_sonic Python | SIMPLE 固定子模块 `e61eeb1dec9678dcc02f97c5ac27ca6a4fe82128` |
| SONIC 权重 | [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC/tree/6733128a3d8a523b1418b06bca3cdf61c8b0987f)，`model_decoder.onnx`、`model_encoder.onnx`、`observation_config.yaml` |
| G1 资产 | [USC-PSI-Lab/SIMPLE](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE/tree/47667c1860fbd55d98d74a0d23d1d5d773a6e9ee)，`robots_g1_sonic.zip` |
| Qwen processor/config | `Qwen/Qwen3-VL-2B-Instruct`，`89644892e4d85e24eaac8bacfd4f463576704203` |

任务权重大小为 6,254,227,440 字节，SHA256 为 `b60c11f84887fc8b6307851c23dc9169cef4c7e292fcaaed706d6254952bc711`，本地和服务器均已核实。原始 `argv.txt`、`run_config.json`、`dataset_statistics.json`、`envs.txt` 随权重保留。

VLM 为 2,131,333,120 参数，动作头为 497,863,904 参数；任务 checkpoint 直接包含两部分权重，不需要重新训练或另外加载通用 VLM 权重。输入是头部左相机 RGB、43 维关节状态及任务语言；输出为 64 维 body token＋14 维手关节，共 78 维。预测长度 30，当前客户端每次执行 24 步，RTC 使用 `test_time`，实际去噪步数为 10。服务实际参数以保存的 `server-info.json` 为准。

## 路径和环境

本地与服务器继续使用原项目根目录，以下均为项目相对路径。当前服务器实测为 A800-SXM4-80GB、驱动 525.105.17、Ubuntu 20.04。

| 路径 | 用途 |
| --- | --- |
| `.venv` | 原有 Python 3.10 / MuJoCo 环境，未安装或更新其中的依赖，未执行 `uv sync` |
| `.venv-sonic` | Python 3.10 增量依赖，由 `run_sonic_eval.py` 接入原环境 |
| `.venv-psi0` | Python 3.11，Psi0 独立推理环境 |
| `.tools/sonic` | GCC 13.4、CMake 3.31、TensorRT 10.9 CUDA 11 和 C++ 依赖 |
| `.tools/cuda-11.8` | 项目既有 CUDA 工具链 |
| `.tools/psi0-source/Psi0` | 固定版本 Psi0 源码 |
| `data/checkpoints/psi0-sonic-box` | 原始任务 checkpoint 与配置 |
| `data/robots/g1_sonic` | G1 SONIC 资产，必须放在数据资源解析器使用的位置 |
| `data/simple-eval/G1WholebodyXMoveBendCarryBoxSonic-v0/dr-level-0` | 原始实例、环境配置与参考动作 |
| `.cache` | 下载包、HF processor、编译及运行缓存 |
| `experiments/reproduction/psi0-sonic` | 持久化日志、指标和视频 |

不执行上游会更改全局 Git 配置或 shell 配置的 `setup_env.sh`，不安装 HOME/bin 编译器替身，不覆盖系统 CUDA。安装过程曾由 micromamba 自动在 `/root/.conda/environments.txt` 登记项目环境；该新增行已移除，其他原有条目保留。实际环境与工具都在项目目录。曾放在 `src/simple/resources/robots/g1_sonic` 的冗余资产副本已与正确目录逐文件比对，清理重复副本并保留 `data/robots/g1_sonic` 的全部 68 个文件。

Python 增量需求见 `sonic-requirements.txt` 和 `psi0-inference-requirements.txt`；实际解析版本保存在实验目录的 `sonic-packages.txt`、`psi0-packages.txt`。Psi0 固定 `torch==2.7.0+cu118`，使用 `uv --no-config pip install` 避开 SIMPLE 默认 CUDA 12 索引。下载不稳定时，先在本地准备 wheel，传至项目 `.cache/psi0-wheelhouse` 后离线安装。

## 已验证运行入口

以下命令从项目根目录执行。工具链、权重、资产需先按上表准备；环境脚本只设置当前 shell，不自动安装依赖。

控制器构建：

```bash
source scripts/reproduction/sonic_env.sh
export SONIC_DOWNLOADS_DIR="$PWD/.cache/psi0-sonic-inputs/build-inputs"
export BUILD_JOBS=4 TRT_CUDA_MAJOR=11
export ZLIB_ROOT="$PWD/.venv-sonic/lib/python3.10/site-packages/cmeel.prefix"
export SONIC_CXX_FLAGS="-I$ZLIB_ROOT/include -pthread"
export CC="$SONIC_TOOLS_DIR/compiler/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$SONIC_TOOLS_DIR/compiler/bin/x86_64-conda-linux-gnu-g++"
export PATH="$SONIC_TOOLS_DIR/compiler/bin:$PATH"
bash scripts/build_sonic_controller.sh
```

控制器专项测试：

```bash
source scripts/reproduction/sonic_env.sh
source "$SONIC_ENV_FILE"
third_party/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/run_tests \
  --gtest_filter='SimLockstep.*:StateLoggerLockstep.*'
```

Psi0 服务（单独终端）：

```bash
source scripts/reproduction/sonic_env.sh
export PYTHONPATH="$PWD/.tools/psi0-source/Psi0/src"
.venv-psi0/bin/python -m psi.deploy.serve_psi0_sonic_http \
  --policy psi0 --host 127.0.0.1 --port 8014 --ckpt-step 40000 \
  --run-dir "$PWD/data/checkpoints/psi0-sonic-box" \
  --rtc-mode test_time --action-exec-horizon 24 --num-inference-steps 10
```

真实观测单步推理检查（不执行动作）：

```bash
source scripts/reproduction/sonic_env.sh
.venv-sonic/bin/python scripts/reproduction/probe_psi0_sonic.py \
  --episode 0 --output experiments/reproduction/psi0-sonic/probe-new
```

闭环评测（每次使用新的输出目录，保留失败结果）：

```bash
source scripts/reproduction/sonic_env.sh
export EVAL_WBC_LOG_DIR="$PWD/experiments/reproduction/psi0-sonic/new-run-logs"
bash scripts/run_eval_wbc.sh simple/G1WholebodyXMoveBendCarryBoxSonic-v0 psi0 \
  --data-dir data/simple-eval/G1WholebodyXMoveBendCarryBoxSonic-v0/dr-level-0 \
  --episode-start 0 --num-episodes 5 --dr-level 0 --sim-mode mujoco \
  --policy-timeout 180 --eval-dir experiments/reproduction/psi0-sonic/new-run
```

以 `replay` 替换 `psi0` 可执行原始参考动作。控制器端口 5556、5557、13579 被占用时入口拒绝启动，不终止未知进程；评测退出后清理本次控制器进程组。共享文件系统上 Python 冷启动实测需数分钟，需结合进程状态和日志判断，不能仅凭短暂无输出判为挂起。

## 实验记录

原始预算保持不变：回放为源轨迹长度＋10 步，Psi0 为 `max(2 × 源长度, 1500)`。未延长预算、替换成功判据或筛掉失败实例。模型采样有随机性，本轮每个实例仅一次，结果是部署 pilot，不是稳定成功率估计。

主批次及结果：

| 运行目录 | 代码提交 | 实例 | 成功 / 总数 | 状态 |
| --- | --- | --- | --- | --- |
| `replay-0` | `b38bdfd351ad1a105257035021780be83ca618f8` | 0 | 0 / 1 | 预算耗尽 |
| `replay-1-4` | `f93a3932ab2bf14115c39674753a407591b29b11` | 1–4 | 0 / 4 | 全部预算耗尽 |
| `psi0-0-1` | `b38bdfd351ad1a105257035021780be83ca618f8` | 0–1 | 0 / 2 | 全部预算耗尽 |
| `psi0-2-4` | `f93a3932ab2bf14115c39674753a407591b29b11` | 2–4 | 0 / 3 | 全部预算耗尽 |

Psi0 总计执行 11,890 步、497 次推理；参考回放总计执行 5,995 步。二者各 0/5 成功，均无模型错误或 lockstep 超时。Psi0 五个实例均未被进度启发式标记为仍在推进；参考回放五个实例被标记为仍有进展，因此这里只能报告既定预算内未完成，不能据此断言更长预算或其他参考方案也会失败。

两段代码之间只增加运行证据落盘、骨盆诊断字段及依赖清单，没有修改物理参数、控制或奖励计算。参考回放 5 个实例均无模型错误或 lockstep 超时，但未通过任务判据。实例 1 箱子最高约 0.257 m；实例 2、3、4 分别抬到约 0.784、0.826、0.824 m，最高任务奖励分别为 0.08、0.22、0.42，仍低于成功门槛。实例 4 最后箱子跌落；不能将短暂放置计为成功。

原始视觉离线诊断（`original-observation-probe`）：从同版本 `psi-data/simple/G1WholebodyXMoveBendCarryBoxSonic-v0.zip` 的 episode 0 取第 0、300、600 帧及对应状态，每帧清空 RTC 历史后推理一次，均返回有限值 `(24,78)`。该文件 SHA256 为 `c8b31cc1ea61a653210278894a33555b735dd33568c4c5ad97ee48ee4871d1af`。仅作公开数据样本上的加载/接口核查，不计入测试集成功率，不是视觉因果对照，也不证明泛化。

| 原始数据帧 | body token RMSE | 手关节 RMSE（rad） |
| --- | --- | --- |
| 0 | 0.01354 | 0.02466 |
| 300 | 0.00813 | 0.00201 |
| 600 | 0.00758 | 0.00324 |

独立旧重力对照：`replay-0-legacy-gravity`，代码 `3da4312f986be2e6f93966099613640d5033a214`，实例 0 仍为 0/1，执行原始 1,064 步预算，无模型错误或通信超时。`legacy-gravity.json` 确认全局重力为 −9.81 m/s²，同时给箱子增加 −0.981 N 向下力。该对照只恢复历史额外外力，没有回退整个上游版本；它没有恢复任务成功，不能把此前的重力修复认定为唯一失败原因。此结果不混入上述主批次的 0/5 统计。

主批次产物一致性检查通过：每个实例的控制确认数等于执行步数；诊断版本每步一条进度记录；保存的动作维度为 78 且全部有限；动作块、推理记录及请求数量一致。汇总见 `validation-summary.json` 和 `artifact-validation.log`。闭环实例 2–4 的推理中位数约 0.57–0.60 秒；采用 lockstep 等待动作，不据此宣称实时控制性能。

本阶段判定：**现成权重的无训练部署与推理已验证，标称任务成功验收未通过。** 继续排查官方视觉输入域，以及参考回放在放置阶段的控制/接触与版本差异；不把当前结果解释为必须训练，也不将这套尚未通过标称验收的组合用于 PHYS 正式排名。Psi0 服务和本次控制器均在实验结束后停止，环境、权重、日志及视频保留在原项目路径。

开发状态：[PR #4](https://github.com/abstcol/SIMPLE/pull/4) 已推送并完成服务器试跑，因标称任务门槛未通过而保留草稿，未合并主分支。上表 SHA 是实际运行版本；后续文档提交没有改动运行代码，不能冒充新的标称成功验证。

证据目录中保留：

- `logs/controller-build-3.log`：最终成功构建；此前两次构建失败日志也保留。
- `logs/controller-lockstep-tests.log`：6 项 lockstep 测试通过。完整测试集另有 FK 测试依赖未发布的 `reference/bones_072925_test/`，未通过验证，不能称全部 C++ 测试通过。
- `server-health.json`、`server-info.json`、`logs/psi0-server.log`：服务协议、严格权重加载和推理日志。
- `probe-0b/result.json`、`observation.png`、`action.npy`：640×360 RGB、43 维状态，返回有限值 `(24,78)` 动作；首次真实观测推理约 16.95 秒。
- 各评测目录的 `summary.json`、`eval_stats.txt`、双目视频；诊断版本另存 `diagnostics.json`、`progress.jsonl`、`action_chunks.npz`，包含请求/动作、箱子位置、骨盆高度和朝向。

`diagnose_legacy_gravity.py` 是单独的回放诊断入口：仅在本进程恢复修复前的物体额外向下力，强制要求 `replay` 和独立输出目录，并保存实际施加外力。该旧行为不能作为 PHYS 基线。

基线继承此前 PR #2 的物体双重重力修复，并非完全未修改的上游物理实现。该差异和 MuJoCo/Isaac 渲染差异都必须在解释标称结果时披露；目前没有足够对照将失败归因于某一项。

## 下一道验收门槛

只有取得可复查的完整标称任务成功，并确认接口、动作语义和成功判据一致后，才把 Psi0＋SONIC 冻结为 PHYS 评测基线。若部署链路正常而任务仍失败，先做有针对性的输入域、参考回放和版本差异诊断；保持无训练约束，不用失败的标称基线直接开展物理鲁棒性排名。
