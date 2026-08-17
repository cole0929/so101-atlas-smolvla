# 从零复现指南（Reproduction Guide）

> 目标读者：有全套硬件（Atlas 200I DK A2 + SO-ARM101 双臂 + 双相机 + 一台 PC）的复现者。
> 本文档回答：「clone 这个仓库后，怎么把 NPU 推理、数字孪生、视觉抓取跑起来？」

---

## 0. 先看：哪些能跑、哪些需要什么

| 模块 | 需要硬件 | 需要板端私有资产 | 能否直接复现 |
|---|---|---|---|
| 文档 / 教程 / 源码阅读 | ❌ | ❌ | ✅ 随时可以 |
| ROS2 数字孪生（RViz2 显示） | 板 + 从臂 | ❌ | ✅ 需自建 ROS2 环境 |
| MoveIt2 真实控制 | 板 + 从臂 | ❌ | ✅ 需自建环境 |
| SmolVLA NPU 推理 | 板 + 双臂 + 双相机 | **模型检查点 + 数据集** | ⚠️ 需训练或索取模型 |
| 视觉引导抓取 | 全套 | **标定数据**（可自采） | ⚠️ 需先完成标定 |

> **核心结论**：代码、URDF、ROS2 包、教程都在仓库里，clone 即可读。
> NPU 推理和抓取依赖「训练好的模型」——模型检查点（约 8.5GB）不在仓库，
> 需要：① 按本文档 §4 从零训练，或 ② 向原作者索取部署产物。

---

## 1. 硬件清单

| 设备 | 型号 / 标识 | 用途 |
|---|---|---|
| 开发板 | Atlas 200I DK A2（aarch64, Ubuntu 22.04） | 板端大脑，`192.168.0.2` / `davinci-mini` |
| NPU | Ascend 310B1 | SmolVLA 推理 |
| Follower 从臂 | SO-ARM101（USB 序列号 `5C82108953`） | 执行 |
| Leader 主臂 | SO-ARM101（USB 序列号 `5C4C123788`） | 示教采集 |
| 前置相机 | USB PID `9221`（eye-to-hand，俯视） | 全局定位 |
| 手眼相机 | USB PID `9005`（eye-in-hand，夹爪上） | 精定位 |
| 训练服务器 | NVIDIA A100（可选，用于自己训练） | 训练阶段 |

## 2. 环境准备（板端）

### 2.1 基础环境

```bash
# 系统：Ubuntu 22.04 aarch64
# 板端三个独立 Python 环境（关键！互不污染）：

/opt/lerobot061            # LeRobot 数据采集/遥操作环境（PyTorch CPU）
/opt/smolvla_npu_test      # NPU 转换与推理实验环境（ONNX 导出）
/opt/cann_py310_deps       # ATC 工具 Python 3.10 依赖（PYTHONPATH 注入）
```

### 2.2 LeRobot 环境（`/opt/lerobot061`）

```bash
python -m venv /opt/lerobot061
/opt/lerobot061/bin/pip install lerobot==0.6.1
# 补 SmolVLA CPU 加载依赖：
/opt/lerobot061/bin/pip install transformers==5.5.4 accelerate==1.14 tokenizers num2words
```

### 2.3 CANN 与 NPU 环境

```bash
# CANN 7.0.RC1（华为官方安装包，需在昇腾社区下载）
# ATC 标准环境：
. /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/opt/cann_py310_deps:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe
```

### 2.4 ROS2 Humble（板端仅 ros-base）

```bash
# 板端：ros-humble-ros-base + rmw_zenoh_cpp
# 详细步骤见 docs/Atlas_ROS2_SO101部署与使用技术文档_2026-08-15.md
source /root/ros2_humble_env.sh
```

## 3. 环境准备（PC / WSL2）

```bash
# Windows 11 + WSL2 Ubuntu 22.04
# ROS2 Humble Desktop（含 RViz2、Gazebo 11）
# 通信：rmw_zenoh_cpp，两端 ROS_DOMAIN_ID=42
# 详细步骤见 docs/Atlas_ROS2_SO101部署与使用技术文档_2026-08-15.md
```

## 4. 复现路线 A：直接部署已训练模型（最快）

1. 向原作者索取板端部署产物：4 个 OM 文件（约 1.2GB）+ 运行时脚本
2. 放到板端 `/root/lerobot_project/`（脚本已在本仓库根目录）
3. 按 `docs/Atlas_SmolVLA_NPU部署与实机推理技术开发文档_2026-08-14.md` §7 验证

## 5. 复现路线 B：从零训练 + 部署（完整复现）

### 5.1 采集数据（LeRobot）

```bash
# 板端
cd /root/lerobot_project
/opt/lerobot061/bin/python 03_teleoperate_with_rviz.py   # 遥操作（先在 RViz 看状态）
# 采集「抓黄块放进黑盒再回位」任务，目标：20Hz / 50 episodes / ~8453 帧
# 双相机 640×480，前置 MJPG 压缩、手眼降分辨率（避免 USB 带宽不足）
```

### 5.2 训练（A100）

```bash
# 模仿学习 20K steps，产出 020000 检查点（约 8.5GB）
# 参考 docs/Atlas_SmolVLA_开发记录与后续计划.md
```

### 5.3 NPU 迁移（板端，按 `npu_tests/` 顺序）

```bash
# 分步验证：每步都有 golden case 对比
01  matmul 冒烟        # 验证 ATC+pyACL 可用（0.21ms，误差 7.8e-4）
03  state_proj 导出    # 6→32 补零 → 1×960
05  vision 导出        # 1×3×512×512 → 1×64×960
08  prefix 导出        # 16 层 KV prefill
10  denoise 导出       # 动作去噪
13  全模型串联          # vision→prefix→denoise×5→Euler 积分→50×6
```

### 5.4 实机验证

```bash
# 板端
/opt/smolvla_npu_test/bin/python 14_smolvla_atlas_live.py --dry-run        # 只验证
/opt/smolvla_npu_test/bin/python 14_smolvla_atlas_live.py --motion-key ENABLE_ATLAS_MOTION  # 实机
/opt/smolvla_npu_test/bin/python 15_smolvla_atlas_evaluate.py               # 成功率评估
```

> ⚠️ 安全：observe 模式默认不发动作；运动必须显式 `--motion-key`；
> 主体 10°、夹爪 20° 相对限幅 + NaN/Inf 拦截 + 网页 SAFE STOP；
> 软件停止不能替代物理断电；运行时工作区不得有手、脸、线缆。

## 6. 复现数字孪生（ROS2）

### 6.1 编译工作空间（WSL）

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --merge-install --symlink-install
source install/setup.bash
```

### 6.2 启动数字孪生

```bash
# 板端：发布真实关节状态
systemctl start so101-rviz-bridge.service

# WSL：一键启动（含边界/禁入区/角度标注叠加层）
source ~/ros2_humble_env.sh
ros2 daemon stop        # Zenoh 环境必需！
ros2 launch so101_visualization trajectory_viz.launch.py
```

### 6.3 验证数据流

```bash
ros2 topic list
ros2 topic hz /joint_states
ros2 topic echo /joint_states --once
```

## 7. 复现视觉抓取

### 7.1 标定（板端，按顺序）

```bash
cd /root/lerobot_project
/opt/lerobot061/bin/python 16_capture_calibration.py        # 采集手眼标定图（15 姿态）
/opt/lerobot061/bin/python 18_capture_front_intrinsics.py   # 前置内参（18 张多角度）
/opt/lerobot061/bin/python 17_handeye_calibrate.py          # 手眼标定
# 前置外参：几何法（棋盘格平放已知位置 + solvePnP + 反投影自检）
# 结果写入 calib_out/（本仓库 calib_out/ 是原作者的标定结果，复现需重新标定）
```

### 7.2 抓取

```bash
/opt/lerobot061/bin/python 20_grasp_block.py --dry-run    # 先看计划
/opt/lerobot061/bin/python 20_grasp_block.py --handeye-offset-y +0.05 --jaw-clearance 0.04
```

> 关键参数：`--jaw-clearance`（下爪位置）、`--handeye-offset-y`（手眼 y 补偿）、
> `--grasp-height`、`--approach-height`、`--max-relative-target`（限幅）。
> 详见 docs/2026-08-16_物块定位与抓取_工作记录.md §4。

## 8. 常见问题（踩坑速查）

| 现象 | 解决 |
|---|---|
| WSL `ros2 topic echo` 报 xmlrpc Fault | `ros2 daemon stop` |
| 板端 ros2 命令找不到 | `source ~/ros2_humble_env.sh` |
| 双相机读失败 | 前置 MJPG、手眼降分辨率 |
| 相机节点号不对 | 按 USB PID 动态识别（9221 前置 / 9005 手眼） |
| RViz 黑屏 | VcXsrv / `LIBGL_ALWAYS_SOFTWARE=1` |
| RViz 模型闪烁 | retime 重打时间戳（桥接自带） |
| ATC 报 Python 版本冲突 | 用 `/opt/cann_py310_deps` PYTHONPATH |
| state OM 报输入尺寸错 | 6→32 补零 |
| 训练环境被污染 | 三个独立 venv，不要混用 |

## 9. 无法从仓库直接获得的东西

| 资产 | 位置 | 获取方式 |
|---|---|---|
| 训练好的模型检查点（8.5GB） | 板端 `/root/lerobot_project/models/` | 训练 or 向作者索取 |
| 4 个 OM 部署文件（1.2GB） | 板端 | 转换 or 向作者索取 |
| 数据集（50 episodes） | 板端 | 重新采集 |
| 标定结果 | 板端 + 本仓库 calib_out/ | 重新标定（推荐，因相机/安装位置不同） |
| SO-ARM101 官方资料（URDF 之外的图纸/视频） | 厂商提供 | 联系 SO-ARM101 官方获取 |
