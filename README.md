# Atlas SO-ARM101 具身智能项目

> 基于开源六轴机械臂 **SO-ARM101** 的具身智能完整闭环：
> **LeRobot 数据采集 → A100 训练 SmolVLA → 昇腾 NPU 板端部署 → ROS2 数字孪生 → 视觉引导抓取**

[![ROS 2](https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros)](https://docs.ros.org/en/humble/)
[![Ascend](https://img.shields.io/badge/Ascend-310B1-00A1E9)](https://www.hiascend.com/)
[![SmolVLA](https://img.shields.io/badge/SmolVLA-0.5B-FF6F00)](https://github.com/HuggingFaceRL/SmolVLA)
[![Python](https://img.shields.io/badge/Python-3.10%2F3.12-3776AB?logo=python)](https://www.python.org/)

---

## 项目简介

本项目让一台华为 Atlas 200I DK A2 开发板成为 SO-ARM101 机械臂的「边缘大脑」，实现**不依赖 PC** 的完整闭环：

```text
双相机采集 → 六关节状态读取 → SmolVLA 视觉/语言/状态融合推理（NPU）
→ 生成 50×6 动作块 → SO-101 从臂执行 → 网页仪表盘监控
```

同时搭建了 **ROS2 数字孪生**（RViz2 实时虚实同步 + MoveIt2 真实控制）与
**视觉引导抓取**（双相机标定 → 物块定位 → IK 解算 → 夹取）两条辅助线。

## 系统架构

```text
┌────────────────────── Atlas 200I DK A2（开发板）──────────────────────┐
│  双相机(前置9221/手眼9005) ──> SmolVLA ──> 50×6 动作块 ──> 从臂执行       │
│  NPU: Ascend 310B1 · CANN 7.0.RC1 · pyACL                               │
│  关节状态 ──> joint bridge ──> /joint_states ──> Zenoh 路由器(7447)      │
└────────────────────────────────────────────────────────────────────────┘
                              │ rmw_zenoh_cpp · ROS_DOMAIN_ID=42
┌────────────────────── Windows PC（WSL2 Ubuntu 22.04）──────────────────┐
│  RViz2 数字孪生（URDF + TF + Marker 叠加层）                             │
│  MoveIt2 运动规划 ──> command bridge ──> 从臂电机                        │
└────────────────────────────────────────────────────────────────────────┘
```

## 功能特性

| 模块 | 说明 | 关键文件 |
|---|---|---|
| 🤖 SmolVLA NPU 部署 | 模型拆 4 子图（vision/state_proj/prefix/denoise），ONNX→OM→pyACL，板端实机推理 ~918ms/次 | `npu_tests/`、`14_smolvla_atlas_live.py` |
| 📡 ROS2 数字孪生 | 真实关节角 → Zenoh 跨机 → RViz2 实时渲染，含工作区边界/禁入区/角度标注叠加层 | `ros2_ws/src/so101_visualization/` |
| 🎯 MoveIt2 真实控制 | RViz 规划 → FollowJointTrajectory → 板端 executor → 串口驱动从臂 | `moveit_real_arm_control/` |
| 📷 视觉抓取 | 前置/手眼双相机标定、物块定位、数值 IK、限幅插值执行 | `16~21_*.py`、`ik_solver.py` |
| 📚 零基础教程 | 从硬件到抓取的完整教学（面试讲解 + 实操），见 `docs/教程/` | `docs/教程/` |

## 目录结构

```text
robot_arm_atlas/
├── README.md                 # 仓库门面（本文档）
├── REPRODUCE.md              # 从零复现指南（有硬件者先读）
├── LICENSE                   # MIT License
├── docs/                     # 技术文档 + 零基础教程（docs/教程/）
├── ros2_ws/src/              # ROS2 工作空间源码
│   ├── so101_description/    #   官方 URDF + STL 模型
│   ├── so101_visualization/  #   数字孪生节点（含叠加层）
│   └── so101_moveit_config/  #   MoveIt2 配置
├── moveit_real_arm_control/  # MoveIt2 真实机械臂控制工程
├── ros2/                     # ROS2 部署脚本、systemd 服务、Zenoh 配置
├── npu_tests/                # NPU 迁移分步验证（01~13）
├── 14_smolvla_atlas_live.py  # 板端实机推理
├── 15_smolvla_atlas_evaluate.py  # 实机成功率评估
├── 16~21_*.py                # 标定与抓取系列
├── ik_solver.py              # 数值 IK（阻尼最小二乘）
├── calib_data/ calib_out/    # 标定原始图与结果
├── golden_case/              # CPU 参考输出（NPU 对比验证）
└── live_check/               # 实机验证截图与遥测
```

## 快速开始

> 完整部署文档：`docs/Atlas_SmolVLA_NPU部署与实机推理技术开发文档_2026-08-14.md`
> 与 `docs/Atlas_ROS2_SO101部署与使用技术文档_2026-08-15.md`
>
> **有全套硬件想复现？先读 [`REPRODUCE.md`](REPRODUCE.md)**（从零复现指南）。
> 零基础学习者从 `docs/教程/README.md` 开始。

### 板端（Atlas 200I DK A2）

```bash
# NPU 实机推理
cd /root/lerobot_project
/opt/smolvla_npu_test/bin/python 14_smolvla_atlas_live.py --motion-key ENABLE_ATLAS_MOTION

# ROS2 只读桥接（发布关节状态到数字孪生）
systemctl start so101-rviz-bridge.service
```

### PC 端（WSL2）

```bash
source ~/ros2_humble_env.sh
ros2 daemon stop                      # Zenoh 环境必需
ros2 launch so101_visualization trajectory_viz.launch.py
```

### 视觉抓取（板端）

```bash
/opt/lerobot061/bin/python 20_grasp_block.py --dry-run   # 先看计划
/opt/lerobot061/bin/python 20_grasp_block.py --handeye-offset-y +0.05 --jaw-clearance 0.04
```

## 硬件清单

| 设备 | 型号/标识 |
|---|---|
| 开发板 | Atlas 200I DK A2（aarch64, Ubuntu 22.04, `192.168.0.2` / `davinci-mini`） |
| NPU | Ascend 310B1（CANN 7.0.RC1） |
| Follower 从臂 | SO-ARM101（USB 序列号 `5C82108953`） |
| Leader 主臂 | SO-ARM101（USB 序列号 `5C4C123788`，示教用） |
| 前置相机 | USB PID `9221`（eye-to-hand，俯视工作区） |
| 手眼相机 | USB PID `9005`（eye-in-hand，夹爪上） |
| 训练服务器 | NVIDIA A100（训练阶段） |

## 致谢

- [SO-ARM101](https://github.com/SCUT-ZN/SO-ARM101) 开源六轴机械臂（官方资料与 LeRobot 源码由厂商提供，不包含在本仓库中）
- [LeRobot](https://github.com/huggingface/lerobot) 机器人学习框架
- [SmolVLA](https://github.com/HuggingFaceRL/SmolVLA) 视觉-语言-动作模型
- 华为 [CANN](https://www.hiascend.com/) 昇腾计算框架

## License

本项目代码采用 [MIT License](LICENSE)，仅供学习研究使用。
SO-ARM101 官方资料、LeRobot、SmolVLA 版权归各自所有者。
