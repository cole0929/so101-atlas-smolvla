# Atlas 200I DK A2 + SO-ARM101 + SmolVLA 开发记录与后续计划

> 状态基准日期：2026-08-14  
> 文档位置：`F:\robot_arm_atlas\docs\Atlas_SmolVLA_开发记录与后续计划.md`  
> 板端项目：`/root/lerobot_project`  
> 说明：本文只记录本项目已经实际完成、现场核对过的内容，以及下一阶段的实施顺序。它不是泛化后的网络教程。

## 1. 项目目标

本项目的目标是让华为 Atlas 200I DK A2 成为 SO-ARM101 机器人侧控制节点，逐步完成以下闭环：

```text
SO-ARM101 Leader 示教
        ↓
Atlas 读取双臂关节和双摄像头
        ↓
LeRobot 记录数据集
        ↓
Atlas 清洗数据
        ↓
数据上传 A100 云服务器训练 SmolVLA
        ↓
模型回传 Atlas
        ↓
Atlas 本地推理并控制 Follower
```

截至本文日期，前六步已完成。最后一项“Atlas NPU 本地执行 SmolVLA 并安全控制机械臂”尚未完成。

## 2. 当前完成状态总览

| 模块 | 当前状态 | 结论 |
|---|---|---|
| PC 与 Atlas 通信 | 已完成 | Type-C 网络，SSH 地址 `192.168.0.2` |
| Atlas 系统与 LeRobot | 已完成 | Ubuntu 22.04/aarch64，LeRobot 0.6.1 |
| Leader/Follower 识别 | 已完成 | 按 USB 序列号识别，不依赖动态 tty 编号 |
| 双机械臂校准 | 已完成 | 校准文件已保存在板端项目目录 |
| 无摄像头遥操作 | 已完成 | 30 Hz 控制程序可运行 |
| 双摄像头遥操作 | 已完成 | 双路取帧与浏览器预览可运行 |
| 数据采集 | 已完成 | 50 episodes、5 条同义任务指令 |
| 空 episode 异常修复 | 已完成 | 空缓冲不再调用 `save_episode()` |
| 数据清洗 | 已完成 | 50 episodes、8453 frames，退出码 0 |
| A100 云训练 | 已完成 | 20K steps，10 个检查点 |
| 训练结果备份到 PC | 已完成 | 完整训练包约 13 GB，SHA256 已校验 |
| 推理检查点部署到 Atlas | 已完成 | 10 个检查点，共约 8.5 GB |
| Atlas CPU 加载模型验证 | 已完成 | 020000 检查点无机械臂 dry-run 通过，50×6 输出有限且可重复 |
| Atlas `torch_npu` 环境 | 未完成 | 当前是 CPU 版 PyTorch，无 `torch_npu` |
| SmolVLA NPU 推理 | 未完成 | 当前权重不是 OM 模型，不能直接调用 NPU |
| Atlas ONNX→OM→pyACL 链路 | 已完成 | Ascend310B1 基础样例和真实 SmolVLA `state_proj` 子图均通过 |
| 实机自主抓取 | 未完成 | 必须通过离线对齐和安全测试后再做 |

## 3. 实际硬件与系统环境

### 3.1 Atlas

| 项目 | 实际值 |
|---|---|
| 开发板 | Huawei Atlas 200I DK A2 |
| 主机名 | `davinci-mini` |
| CPU 架构 | `aarch64` |
| 操作系统 | Ubuntu 22.04 LTS |
| 内核 | Linux 5.10.0+ |
| NPU | Ascend 310B1 |
| NPU 管理工具 | `npu-smi 23.0.rc3` |
| NPU 健康状态 | OK |
| CANN Toolkit | 7.0.RC1，内部版本 `7.0.0.5.242` |
| 系统盘 | 117 GB |
| 当前已用/可用 | 约 33 GB / 80 GB |

当前 NPU 检查命令：

```bash
npu-smi info
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

### 3.2 当前 LeRobot Python 环境

```text
Python       3.12.11
LeRobot      0.6.1
PyTorch      2.7.1+cpu
torchvision  0.22.1
虚拟环境      /opt/lerobot061
```

当前关键事实：

```text
hasattr(torch, "npu") == False
torch_npu 未安装
```

所以 `/opt/lerobot061` 目前适合校准、遥操作、摄像头、数据采集和数据清洗，但不能把 SmolVLA 放到 NPU 上运行。

### 3.3 机械臂与相机

机械臂控制板均为 CH343，系统节点可能随插拔变化，所以代码按 USB 序列号识别：

| 角色 | USB 序列号 | 当前常见节点 |
|---|---|---|
| Leader | `5C4C123788` | `/dev/ttyACM0` |
| Follower | `5C82108953` | `/dev/ttyACM1` |

相机按照 USB 产品 PID 识别并实际试读可用节点：

| 角色 | USB PID | 当前节点 | 工作格式 |
|---|---|---|---|
| 手眼相机 | `9005` | `/dev/video0` | YUYV 640×480@30 |
| 前置相机 | `9221` | `/dev/video2` | MJPG 640×480@30 |

`/dev/video1` 和 `/dev/video3` 是 metadata 节点，不能当作图像节点。

## 4. 网络和登录方式

PC 通过 Type-C USB 网络连接 Atlas：

```text
Atlas: 192.168.0.2
SSH:   root@192.168.0.2:22
```

Windows PowerShell 登录命令：

```powershell
ssh -i C:\Users\98384\.ssh\codex_robot_arm_ed25519 root@192.168.0.2
```

MobaXterm 也可直接建立 `root@192.168.0.2` 会话。出现 `Software caused connection abort` 时重新连接即可。

本项目不再修改家庭路由器配置。软件和模型优先采用 PC 中转、SCP 传输，避免影响手机和其他设备上网。

## 5. 板端目录和程序

实际项目目录：

```text
/root/lerobot_project/
├── 01_calibrate_leader.py
├── 02_calibrate_follower.py
├── 03_teleoperate_without_cameras.py
├── 04_teleoperate_with_cameras.py
├── 05_record_dataset.py
├── 06_clean_dataset.py
├── atlas_runner.py
├── lerobot_record_atlas_patched.py
├── cache/
├── lerobot_home/
├── datasets/
└── models/
```

PC 上的同步开发副本：

```text
F:\robot_arm\atlas_lerobot\
```

### 5.1 `atlas_runner.py`

公共启动器完成以下工作：

- 固定调用 `/opt/lerobot061/bin/python`；
- 设置 `HF_HOME`、`HF_HUB_CACHE`、`HF_LEROBOT_HOME`；
- 按 USB 序列号寻找 Leader/Follower；
- 按 USB PID 寻找相机，并用 OpenCV 试读以跳过 metadata 节点；
- 输出实际执行命令和安全提示。

### 5.2 校准

```bash
cd /root/lerobot_project
/opt/lerobot061/bin/python 01_calibrate_leader.py
/opt/lerobot061/bin/python 02_calibrate_follower.py
```

校准文件：

```text
/root/lerobot_project/lerobot_home/calibration/robots/so101_follower/my_awesome_follower_arm.json
/root/lerobot_project/lerobot_home/calibration/teleoperators/so101_leader/my_awesome_leader_arm.json
```

目录内还存在旧类型名 `so_follower`、`so_leader` 的兼容副本。当前程序使用 `so101_follower` 和 `so101_leader`。

### 5.3 无摄像头遥操作

```bash
cd /root/lerobot_project
/opt/lerobot061/bin/python 03_teleoperate_without_cameras.py
```

程序以 30 Hz 读取 Leader 六个关节并控制 Follower。首次启动必须让两臂姿态接近，工作区清空，并准备按 `Ctrl+C`。

### 5.4 双摄像头遥操作与画面预览

```bash
cd /root/lerobot_project
/opt/lerobot061/bin/python 04_teleoperate_with_cameras.py
```

PC 浏览器访问：

```text
http://192.168.0.2:8080
```

控制循环保持约 30 Hz，浏览器预览约 10 FPS。画面预览使用板端 MJPEG HTTP 服务，不依赖 X11、OpenCV GUI 或 Rerun。

## 6. 数据采集实施

### 6.1 采集目标

任务动作定义：

```text
夹取黄色物块 → 放入黑色盒子 → 松开夹爪 → 机械臂返回初始位置
```

为了让 SmolVLA 接触不同语言表达，同一动作使用 5 条语义等价的英文指令，每条录制 10 个 episode：

1. `Pick up the yellow block, place it inside the black box, release it, then return the arm to its starting position.`
2. `Grasp the yellow block, put it into the black box, let go of it, and move the arm back to its initial pose.`
3. `Move the yellow object into the black container, release the object, then return the robot arm to the start position.`
4. `Lift the yellow cube, place it in the black box, open the gripper, and bring the arm back to its home pose.`
5. `Put the yellow item inside the black box, release it there, then retract the arm to its original position.`

注意：这种做法只能增强同一任务范围内的语言鲁棒性，不能单凭 5 个同义句证明模型具备开放世界语义泛化能力。

### 6.2 采集配置

| 参数 | 值 |
|---|---|
| 原始 repo_id | `admin/smolvla_yellow_block_atlas_v2` |
| episode 数 | 50 |
| 每条任务指令 | 10 episodes |
| 数据集 FPS | 20 |
| 单轮最长时间 | 30 s |
| 环境复位时间 | 20 s |
| 图像分辨率 | 两路 640×480 |
| 视频编码 | H.264、CRF 23、GOP 20、ultrafast |
| Hub 上传 | false |
| 声音提示 | false |

运行命令：

```bash
cd /root/lerobot_project
/opt/lerobot061/bin/python 05_record_dataset.py
```

脚本会读取已有 `total_episodes` 并从已完成位置继续，不会盲目覆盖已有数据。

### 6.3 录制期间解决的问题

#### 双摄像头带宽不足

两路都使用未压缩 YUYV 640×480@30 时，同一 USB 2.0 总线无法稳定承载，表现为：

```text
OpenCVCamera read failed (status=False)
Not enough bandwidth for new device state
```

解决方案：

```text
front   → MJPG 640×480@30
handeye → YUYV 640×480@30
```

#### 实际录制循环低于 30 Hz

板端同时读取双摄像头、控制机械臂和编码视频时曾只有约 21.5 Hz，因此正式数据集 FPS 改为 20，避免把 30 FPS 写进元数据但实际持续掉帧。

#### 空 episode 导致保存失败

按键过快或新一轮忘记执行动作时，可能在第一帧进入缓冲区前结束，原版代码会报：

```text
You must add one or several frames with `add_frame` before calling `add_episode`.
```

`lerobot_record_atlas_patched.py` 已增加判断：如果 `dataset.has_pending_frames()` 为假，则清空空缓冲并重试同一 episode，不再调用 `save_episode()`。

#### 非致命警告

- `spd-say` 不存在：已通过 `--play_sounds=false` 规避；
- SVT/编码器线程优先级设置失败：不影响产物，后续已使用 H.264/PyAV 流程；
- 终端中的 2022 时间戳：开发板系统时钟未同步，不代表文件来自 2022 年，后续需要配置本地 NTP 或每次联网后同步时间。

## 7. 数据清洗实施

原始数据集：

```text
/root/lerobot_project/datasets/admin/smolvla_yellow_block_atlas_v2
```

清洗结果：

```text
/root/lerobot_project/datasets/admin/smolvla_yellow_block_atlas_v2_clean
```

运行命令：

```bash
cd /root/lerobot_project
/opt/lerobot061/bin/python 06_clean_dataset.py
```

清洗不是修改原数据，而是新建一个数据集。算法对每个 episode：

1. 读取 `action` 和 `observation.state`，检查 NaN/Inf；
2. 用开头最多 15 帧动作中位数作为静止基线；
3. 任一关节相对基线位移大于 3°，判为运动候选；
4. 在连续 10 帧窗口内至少 5 帧满足运动条件，判定真实起动；
5. 起动点前保留 10 帧上下文；
6. 用同一个裁剪范围同步重建关节数据、front 视频和 handeye 视频；
7. 使用 PyAV 读取并重新编码视频；
8. 写出新的 metadata、parquet、视频和裁剪报告。

清洗成功标志：

```text
DATASET_CLEAN_FINISHED
episodes=50
frames=8453
exit code=0
```

清洗后实际数据：

| 项目 | 值 |
|---|---|
| codebase_version | v3.0 |
| episodes | 50 |
| frames | 8453 |
| tasks | 5 |
| fps | 20 |
| action | 6 维 float32 |
| observation.state | 6 维 float32 |
| front | 480×640×3 H.264 |
| handeye | 480×640×3 H.264 |
| 数据集体积 | 约 182 MB |

两个视频文件约为：

```text
front:   89,592,357 bytes
handeye: 99,636,545 bytes
```

裁剪报告：

```text
/root/lerobot_project/datasets/admin/smolvla_yellow_block_atlas_v2_clean_trim_points.csv
```

## 8. 云服务器训练

### 8.1 数据与模型目录

云服务器训练根目录：

```text
/root/smolvla_training
```

关键目录：

```text
/root/smolvla_training/datasets/admin/smolvla_yellow_block_atlas_v2_clean
/root/smolvla_training/models/smolvla_base_so101_two_cameras
/root/smolvla_training/outputs/smolvla_yellow_block_atlas_v2_clean_a100
```

训练脚本 PC 副本：

```text
F:\robot_arm\smolvla_project\train_atlas_dataset_a100.py
```

### 8.2 训练参数

| 参数 | 值 |
|---|---|
| GPU | A100 |
| steps | 20,000 |
| batch size | 16 |
| workers | 4 |
| prefetch factor | 2 |
| save frequency | 2,000 steps |
| log frequency | 20 |
| AMP | false |
| video backend | pyav |
| WandB | disabled |
| 训练恢复 | 支持从 `checkpoints/last` 继续 |

AMP 关闭的原因是当前组合出现 BF16 gradient unscale 问题；不是因为 A100 不支持混合精度。当前训练已稳定完成，不能为了显存占用更高而随意改变已验证配置。

### 8.3 模型结构事实

训练配置为 SmolVLA：

```text
VLM backbone: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
输入: 6维状态 + front图像 + handeye图像 + 任务文本
输出: 50步 × 6维动作块
freeze_vision_encoder: true
train_expert_only: true
train_state_proj: true
```

所以“总模型规模约 4.5 亿参数”和“实际训练参数约 1 亿”并不矛盾：视觉/语言主干大部分被冻结，优化器只更新动作专家和状态投影等部分。推理仍需加载完整模型权重。

### 8.4 训练结果

训练完成 20K steps，共保存：

```text
002000  004000  006000  008000  010000
012000  014000  016000  018000  020000
```

完整训练目录约 13 GB，是因为每个检查点同时包含：

- 约 0.85 GB 的 `pretrained_model`；
- 约 0.39 GB 的优化器/训练状态；
- 配置和日志。

单次推理只加载一个 `pretrained_model`，不是加载 13 GB。

## 9. PC 备份和校验

完整训练包：

```text
F:\robot_arm\transfer_cache\smolvla_yellow_block_atlas_v2_clean_a100_bundle.tar
```

实际大小：

```text
13,196,257,280 bytes（约 12.29 GiB）
```

SHA256：

```text
09d64c24ae026878eb8a1e2fce2d16e00fea7ac347c012bccbe23f7e67cdb846
```

仅推理检查点包：

```text
F:\robot_arm\transfer_cache\smolvla_yellow_block_atlas_v2_clean_a100_inference_checkpoints.tar
```

实际大小：

```text
9,067,459,584 bytes（约 8.45 GiB）
```

其 SHA256：

```text
a8799252bde3bb59ee07b62ce8126d71d9179d4e6b1d0e679e13ebf966206a25
```

完整包用于备份和继续训练；推理包用于部署。云服务器上的原训练目录和归档没有删除。

## 10. 模型部署到 Atlas

板端模型根目录：

```text
/root/lerobot_project/models/smolvla_yellow_block_atlas_v2_clean_a100
```

目录结构：

```text
smolvla_yellow_block_atlas_v2_clean_a100/
├── 002000/pretrained_model/
├── 004000/pretrained_model/
├── 006000/pretrained_model/
├── 008000/pretrained_model/
├── 010000/pretrained_model/
├── 012000/pretrained_model/
├── 014000/pretrained_model/
├── 016000/pretrained_model/
├── 018000/pretrained_model/
├── 020000/pretrained_model/
└── latest -> 020000/pretrained_model
```

每个 `model.safetensors` 的实际大小：

```text
906,712,520 bytes
```

每个推理目录还包含：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_preprocessor_step_5_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
train_config.json
```

10 个检查点的 `model.safetensors` 和 `config.json` 均已逐项验证。临时上传到板端的 8.45 GB tar 包在验证后已删除，模型目录占约 8.5 GB。

默认路径：

```text
/root/lerobot_project/models/smolvla_yellow_block_atlas_v2_clean_a100/latest
```

它目前指向：

```text
/root/lerobot_project/models/smolvla_yellow_block_atlas_v2_clean_a100/020000/pretrained_model
```

`latest` 只表示训练步数最后，不表示真实抓取成功率一定最高。最终部署前必须比较多个检查点。

## 11. 当前不能直接推理的原因

模型已回传不等于 NPU 推理已完成，原因有四个：

1. 当前模型是 PyTorch `safetensors`，不是 Ascend 可直接加载的 `.om`；
2. 模型 `config.json` 中设备仍记录为 `cuda`，运行时必须正确覆盖；
3. `/opt/lerobot061` 是 `torch 2.7.1+cpu`，没有 `torch_npu`；
4. SmolVLA 不是单一 CNN，而是图像、文本、VLM、状态投影、动作生成和预/后处理组成的复合策略。

当前 CANN 7.0.RC1 与新 PyTorch/LeRobot 存在明显版本断层。华为官方配套表中，CANN 7.0.RC1 对应的公开 `torch_npu` 组合主要是 PyTorch 1.11、2.0.1 或 2.1.0，Python 最高到 3.10；而当前 LeRobot 0.6.1 环境使用 Python 3.12、PyTorch 2.7.1。因此不能在现有环境里直接执行一条 `pip install torch_npu` 命令解决。

相关官方资料：

- [Ascend Extension for PyTorch 版本配套关系](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/quickstart/releasenote/releasenote_0001.html)
- [torch_npu 安装与 wheel ABI 说明](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/configandinstg/instg/insg_0007.html)
- [Atlas 200I DK A2 使用 ATC 将 ONNX 转换为 OM](https://www.hiascend.com/document/detail/zh/Atlas200IDKA2DeveloperKit/23.0.RC2/Application%20Development%20Guide/tmuacop/tmuacop_0013.html)

## 12. 下一步总体原则

下一阶段不能直接连接机械臂试跑。正确顺序是：

```text
检查点筛选
  ↓
CPU 无机械臂加载测试
  ↓
固定输入/输出金样
  ↓
NPU 技术路线验证
  ↓
CUDA/CPU/NPU 输出对齐
  ↓
短时、限速、可急停实机测试
  ↓
正式 rollout 评估
```

任何一步失败，都应停在当前阶段，不要用实机运动代替离线验证。

## 13. 下一步实施计划

### 阶段 1：冻结现状和选择候选检查点

目标：不默认认为 20K 最好。

1. 保留 PC 的完整 13 GB 备份；
2. 不删除 Atlas 的 10 个推理检查点；
3. 先选 `010000`、`014000`、`018000`、`020000` 四个候选；
4. 在 CUDA 电脑或云服务器用同一组离线样本比较动作输出和 loss；
5. 再进行每个检查点至少 10 次真实 rollout，记录成功率，而不是只看训练 loss；
6. 选出真实成功率最高且动作最稳定的版本，再更新 `latest`。

验收产物：

```text
checkpoint_evaluation.csv
包含：checkpoint、离线误差、10次成功数、平均完成时间、异常动作次数
```

### 阶段 2：Atlas CPU 无机械臂离线测试（已于 2026-08-14 完成）

目标：先验证模型目录、tokenizer、预处理器、双图像键名和动作后处理能够在 aarch64 环境加载。

安全要求：拔掉 Follower 动力或至少不打开串口，不调用 `robot.send_action()`。

需要新增脚本：

```text
/root/lerobot_project/07_smolvla_cpu_dry_run.py
```

脚本应完成：

1. 从清洗数据集中读取固定 episode 的一帧；
2. 读取 `front`、`handeye`、6 维 `observation.state` 和任务文本；
3. 从本地 `pretrained_model` 加载 policy；
4. 强制设备为 CPU，不使用配置中的 `cuda`；
5. 执行预处理、一次策略前向和后处理；
6. 输出动作形状、最小值、最大值、是否有 NaN/Inf；
7. 记录加载时间、首帧延迟、后续 10 次平均延迟和峰值内存；
8. 将输入与输出保存为 `.npz`，供 NPU 对齐。

验收标准：

- 模型能够完整加载；
- 输入键名严格为 `observation.images.front`、`observation.images.handeye`、`observation.state`；
- 输出形状符合 50×6 动作块；
- 无 NaN/Inf；
- 不发生 OOM；
- 不连接机械臂也不会打开串口。

实测结果：

```text
checkpoint:              020000/pretrained_model
dataset sample:          episode 0, frame 0
device:                  CPU (torch 2.7.1+cpu, 4 threads)
model load:              45.877 s
first inference:         73.913 s
10-run steady mean:      73.676 s
steady min/max:          73.406 / 73.871 s
peak RSS:                2563.25 MiB
postprocessed shape:     50 × 6
finite:                  true
10-run max output diff:  0.0
result:                  CPU_DRY_RUN_PASS
```

新增脚本和产物：

```text
/root/lerobot_project/07_smolvla_cpu_dry_run.py
/root/lerobot_project/golden_case/cpu_020000_ep000_frame000/
├── raw_input.npz
├── task.txt
├── preprocessed_inputs.npz
├── raw_actions_cpu.npy
├── actions_postprocessed_cpu.npy
└── report.json
```

为满足 SmolVLA CPU 加载，已从 PC 离线补齐 `transformers 5.5.4`、`accelerate 1.14.0`、`tokenizers 0.22.2`、`num2words 0.5.14` 及它们的缺失依赖；`pip check` 通过。PyTorch 仍为 `2.7.1+cpu`，没有安装 `torch_npu`，没有更改 CANN。SmolVLM 本地构造仅补齐了约 4.7 MB 的 config/tokenizer 资源，通过 `load_vlm_weights=false` 避免先加载 2 GB 原始基座权重，再使用 `strict=true` 从 865 MB 微调检查点完整恢复策略参数。

CPU 路线已证明模型、数据键、预处理、前向和反归一化后处理均可用，但约 73.7 秒/动作块的性能不适合实时 rollout。下一步应使用这组固定输入和 CPU 输出建立 CUDA/NPU 对齐，仍不连接机械臂。

### 阶段 3：建立 CUDA 金样

在 A100 或现有 CUDA 电脑上对同一输入生成参考结果：

```text
golden_case/
├── front.npy
├── handeye.npy
├── state.npy
├── task.txt
├── preprocessed_inputs.npz
├── raw_actions_cuda.npy
└── actions_postprocessed_cuda.npy
```

必须保存随机种子和推理配置。SmolVLA 动作生成包含迭代/采样过程，如果随机种子或推理步数不同，输出无法直接比较。

### 阶段 4：验证 NPU 路线，不破坏现有环境

禁止直接改 `/opt/lerobot061`。先创建独立实验环境，例如：

```text
/opt/smolvla_npu_test
```

需要先做版本决策：

#### 方案 A：升级完整 Ascend 软件栈后使用 `torch_npu`

前提：找到同时满足 Atlas 200I DK A2/310B1、aarch64、CANN、Python、PyTorch、torch_npu 的官方配套组合，并确认该组合能承载 SmolVLA 所需算子。

风险：升级顺序涉及固件、驱动、CANN、PyTorch 和 torch_npu，错误组合可能导致 NPU 整体不可用；即使 NPU 张量可运行，LeRobot 0.6.1 与其 Python/PyTorch 下限也可能冲突。

在没有完整备份镜像和恢复方案前，不升级当前生产系统。

#### 方案 B：PyTorch/ONNX → ATC → OM（当前更可控）

板端保留：

- 相机和串口控制；
- 文本分词及无法导出的控制逻辑；
- 数据预处理和动作后处理。

尽可能导出的神经网络子图转换为 ONNX，再用 ATC 生成针对 310B 的 OM 模型，由 ACL/pyACL 调用。

SmolVLA 很可能不能一次性整体导出，需要拆分验证：

1. 图像编码器子图；
2. 状态投影；
3. 文本/VLM 子图；
4. 动作专家或去噪迭代子图；
5. Python 侧拼接和 10 步迭代控制。

第一步只导出最小子图并使用固定 batch=1、固定图像尺寸和固定文本长度。不要一开始尝试把完整控制循环一次转换成 OM。

ATC 的 `--soc_version` 必须用板卡实际支持值确认，不能根据 `npu-smi` 输出自行猜测。官方 Atlas 200I DK A2 示例使用 `Ascend310B4`，但本板 `npu-smi` 显示 310B1，转换前必须执行 `atc --help`、查询芯片支持列表并按当前镜像确认。

### 阶段 5：NPU 基础和子图验收

先做与机器人无关的测试：

1. NPU 矩阵运算或官方样例；
2. 单一图像编码器子图连续执行 100 次；
3. 固定输入下与 CUDA 输出比较；
4. 记录首帧和稳态延迟；
5. 记录 NPU 内存和系统内存；
6. 检查是否有 CPU fallback；
7. 检查每个算子是否真正运行在 NPU。

验收标准：

- 连续运行无崩溃；
- 无 NaN/Inf；
- 无不受支持算子；
- 无隐藏 CPU 回退造成的超高延迟；
- 误差在动作安全阈值内；
- 总推理频率满足控制需求。

#### 2026-08-14 实测进展

已创建独立实验环境 `/opt/smolvla_npu_test`，没有安装 `torch_npu`，没有升级 CANN。CANN 7.0.RC1 的 ATC 实际使用系统 Python 3.10，原镜像缺少编译依赖；已将对应依赖隔离安装到 `/opt/cann_py310_deps`，不覆盖系统 Python。

NPU 基础 MatMul/Add 样例：

```text
soc:                 Ascend310B1
ONNX→OM:             PASS
pyACL executions:    100
mean latency:        0.210 ms
max abs error:       7.813e-4
result:              NPU_SMOKE_PASS
```

真实 `020000` 检查点已训练 `SmolVLA.model.state_proj` 子图：

```text
input/output:        1×32 → 1×960
ONNX→OM:             PASS
pyACL executions:    100
mean latency:        0.218 ms
max abs error:       3.392e-4
mean abs error:      7.912e-5
finite:              true
result:              SMOLVLA_STATE_PROJ_NPU_PASS
```

产物位置：

```text
/root/lerobot_project/npu_tests/matmul/
/root/lerobot_project/npu_tests/state_proj/
/root/lerobot_project/npu_tests/01_build_matmul_onnx.py
/root/lerobot_project/npu_tests/02_run_matmul_om.py
/root/lerobot_project/npu_tests/03_export_smolvla_state_proj.py
/root/lerobot_project/npu_tests/04_run_smolvla_state_proj_om.py
```

这证明板端 ATC、OM、pyACL 和 310B1 执行链路可用，但仅代表策略中的状态投影层已迁移，不代表完整 SmolVLA 已能在 NPU 上 rollout。下一个关键子图是视觉编码器，然后是语言/VLM 前缀和动作专家及 10 步生成循环。

### 阶段 6：完整策略离线对齐

用阶段 3 的固定输入比较：

```text
CUDA raw output
CPU raw output
NPU raw output
CUDA postprocessed action
CPU postprocessed action
NPU postprocessed action
```

重点比较最终反归一化后的六个关节目标，而不是只比较中间 embedding。

建议输出：

```text
max_abs_error
mean_abs_error
每个关节最大误差
50步动作块随时间的误差
```

如果误差会导致明显姿态变化，禁止实机测试。

### 阶段 7：实机推理程序

离线测试全部通过后新增：

```text
/root/lerobot_project/08_smolvla_rollout_atlas.py
```

程序应具备：

- 继续使用 USB 序列号和相机 PID 自动映射；
- 加载选定检查点，而不是硬编码 20K；
- 明确打印推理设备和后端；
- 相机线程只保留最新帧；
- 动作限幅、关节速度限制和异常值拦截；
- 超时保护；
- `Ctrl+C` 清理；
- 物理急停或立即断电方案；
- 初始位姿偏差过大时拒绝启动；
- 可配置 1 秒、3 秒、10 秒的运行上限；
- 默认不保存视频，以先测纯推理性能。

测试顺序：

```text
不通电只打印动作
→ 通电但不发送动作
→ 单步发送
→ 1秒
→ 3秒
→ 10秒
→ 完整任务
```

### 阶段 8：rollout 评估和数据闭环

策略稳定后才开始保存评估数据。数据集名称应以 `rollout_` 开头，例如：

```text
admin/rollout_smolvla_yellow_block_atlas_v1
```

每个候选检查点至少测试 10 次，记录：

- 是否成功夹取；
- 是否放入黑盒；
- 是否可靠松开；
- 是否返回初始位；
- 总耗时；
- 是否碰撞、抖动或超限；
- 语言指令；
- 物块和盒子位置变化。

失败样本应分类，而不是全部直接加入训练：视觉失败、定位失败、夹取失败、释放失败、返回失败、超时、系统异常应分别统计。

## 14. 推荐的最近三项具体工作

下一次开发应严格按以下顺序进行：

1. 编写并运行 `07_smolvla_cpu_dry_run.py`，在不连接机械臂情况下验证板端能否完整加载模型；
2. 在 A100 生成固定输入的 CUDA 金样，与板端 CPU 输出对齐；
3. 单独建立 NPU 实验环境并评估“升级 torch_npu”与“拆分 ONNX/OM”两条路线，不破坏当前可用的采集环境。

当前不应该做的事：

- 不要直接执行未知来源的 `pip install torch_npu`；
- 不要覆盖 `/opt/lerobot061`；
- 不要删除 10 个检查点或 PC 完整备份；
- 不要把 `device=cuda` 的配置直接用于板端；
- 不要在没有离线对齐时连接机械臂跑模型；
- 不要把训练 loss 最低等同于真实任务最好；
- 不要为了“显存占满”调整训练配置；
- 不要修改家庭路由器配置。

## 15. 常用检查命令

### 板端系统

```bash
hostname
uname -a
cat /etc/os-release
df -h /root
npu-smi info
```

### 硬件

```bash
ls -l /dev/ttyACM*
ls -l /dev/video*
python -m serial.tools.list_ports -v
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video2 --list-formats-ext
fuser /dev/ttyACM0 /dev/ttyACM1 /dev/video0 /dev/video2
```

### Python/LeRobot

```bash
/opt/lerobot061/bin/python --version
/opt/lerobot061/bin/python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__)"
/opt/lerobot061/bin/python -c "import importlib.util; print(importlib.util.find_spec('torch_npu'))"
```

### 数据集

```bash
cat /root/lerobot_project/datasets/admin/smolvla_yellow_block_atlas_v2_clean/meta/info.json
du -sh /root/lerobot_project/datasets/admin/smolvla_yellow_block_atlas_v2_clean
find /root/lerobot_project/datasets/admin/smolvla_yellow_block_atlas_v2_clean/videos -type f -ls
cat /root/lerobot_project/clean_dataset.exit
tail -100 /root/lerobot_project/clean_dataset.log
```

### 模型

```bash
MODEL_ROOT=/root/lerobot_project/models/smolvla_yellow_block_atlas_v2_clean_a100
du -sh "$MODEL_ROOT"
readlink -f "$MODEL_ROOT/latest"
find -L "$MODEL_ROOT/latest" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

检查 10 个检查点：

```bash
MODEL_ROOT=/root/lerobot_project/models/smolvla_yellow_block_atlas_v2_clean_a100
for s in 002000 004000 006000 008000 010000 012000 014000 016000 018000 020000; do
  test -s "$MODEL_ROOT/$s/pretrained_model/model.safetensors" && \
  test -s "$MODEL_ROOT/$s/pretrained_model/config.json" && \
  echo "VERIFIED_$s"
done
```

## 16. 安全规则

- 校准 Follower 时必须托住机械臂，防止失去扭矩后下坠；
- 启动遥操作前让 Leader/Follower 姿态接近；
- 初次测试只动一个关节且幅度要小；
- 同一时间只能有一个程序占用串口和相机；
- 关闭浏览器不等于停止机械臂，必须在终端按 `Ctrl+C`；
- 模型第一次实机测试必须限时、限速、限幅；
- 任何 NaN、Inf、输出跳变或频率严重不足都必须中止；
- 软件停止不可靠时立即切断执行器电源；
- 不允许在机械臂工作范围内放置手、脸、线缆或易碎物品；
- NPU 转换和性能测试必须先在机械臂断开的条件下完成。

## 17. 当前项目一句话结论

现在 Atlas 已经能够独立完成 SO-ARM101 校准、双臂遥操作、双摄像头采集和数据清洗；A100 已完成 SmolVLA 训练，全部推理检查点也已安全回传 Atlas。当前真正剩下的核心工作不是“再传一次模型”，而是解决 SmolVLA 与 Ascend 310B1 的执行后端兼容、建立 CUDA/CPU/NPU 输出对齐，并通过安全关卡完成板端实机自主推理。

## 18. 2026-08-14 板端 NPU 推理与实机运动完成记录

本节覆盖并更新上面的阶段计划。Atlas 200I DK A2 已经完成 SmolVLA 全模型拆分部署，PC 不参与推理，只用于传输和维护脚本。

### 18.1 已部署的 NPU 子图

```text
vision.om          双相机视觉编码器与 connector，单相机约 0.187 秒
state_proj.om      32 维补齐状态投影，约 0.0005 秒
prefix_origin.om   16 层语言前缀 KV prefill，约 0.178 秒
denoise_origin.om  单步动作专家去噪，约 0.056 秒
```

完整 10 步 Euler 推理在真实相机测试中为约 `1.14 秒/次`。固定金样整链测试为 `1.365 秒/次`（首次加载和测试框架开销略有差异）。

### 18.2 数值对齐结果

```text
去噪 ONNX wrapper vs LeRobot 原生 denoise_step:
  max_abs_error = 2.38418579e-6

单步 NPU OM vs CPU wrapper:
  max_abs_error  = 4.05311584e-6
  mean_abs_error = 3.86235399e-7

完整 10 步 NPU vs 相同 NPU 前缀的 CPU Euler:
  max_abs_error  = 1.90734863e-6
  mean_abs_error = 1.56539315e-7

完整 NPU vs 原始 CPU 策略（包含视觉精度差异）:
  归一化动作 max_abs_error  = 0.0281903
  归一化动作 mean_abs_error = 0.00190363
  还原角度后的平均绝对误差约 0.043 度
```

所有输出均为有限值，最终动作块维度为 `50×6`。

### 18.3 真实硬件验证

板端脚本：

```text
/root/lerobot_project/14_smolvla_atlas_live.py
```

真实硬件自动映射结果：

```text
Follower: USB serial 5C82108953
front:    USB PID 9221 -> /dev/video0
handeye:  USB PID 9005 -> /dev/video2
```

`observe` 模式已通过：连接真实从臂和两路相机，采集实时观测并在 NPU 上完成推理，没有发送舵机命令。

`single-step` 模式已通过：实时推理首动作经过主体关节最多 `2°`、夹爪最多 `5°` 的显式限幅后发送。关节回读确认肩关节、肘关节和腕部产生对应位移，程序随后正常断开并卸力。这证明机械臂已经由开发板本地 SmolVLA 推理实际驱动。

### 18.4 启动命令

只观察，不允许运动：

```bash
cd /root/lerobot_project
. /usr/local/Ascend/ascend-toolkit/set_env.sh
/opt/smolvla_npu_test/bin/python 14_smolvla_atlas_live.py --mode observe
```

限幅单步（工作区清空且有人看护时）：

```bash
cd /root/lerobot_project
. /usr/local/Ascend/ascend-toolkit/set_env.sh
/opt/smolvla_npu_test/bin/python 14_smolvla_atlas_live.py \
  --mode single-step --motion-key ENABLE_ATLAS_MOTION
```

连续推理模式已经部署但尚未长时间实机评估。必须按 `1 秒 -> 3 秒 -> 10 秒 -> 完整任务` 逐级放行：

```bash
cd /root/lerobot_project
. /usr/local/Ascend/ascend-toolkit/set_env.sh
/opt/smolvla_npu_test/bin/python 14_smolvla_atlas_live.py \
  --mode rollout --motion-key ENABLE_ATLAS_MOTION --duration 1
```

### 18.5 当前结论

Atlas 已经不依赖 PC 完成真实双相机、实时关节状态、SmolVLA 视觉/语言/动作专家 NPU 推理及从臂动作下发。当前状态从“等待部署”更新为“板端推理已驱动机械臂，下一阶段为有人看护的连续 rollout 成功率评估”。

### 18.6 Atlas 网页仪表盘

`14_smolvla_atlas_live.py` 已加入由开发板直接提供的网页仪表盘，默认监听：

```text
http://192.168.0.2:8080
```

仪表盘包含：

- 前置相机和手眼相机实时画面；
- 六关节实际位置、下发目标和最近 20 秒历史曲线；
- 观察刷新率、NPU 完整推理耗时和当前运行状态；
- `SAFE STOP` 网页安全停止按钮；
- 每次运行目录中的 `telemetry.csv` 遥测日志。

仪表盘默认随 `observe`、`single-step` 和 `rollout` 启动。`observe` 推理结束后默认保留网页 30 秒，可使用 `--dashboard-hold` 调整。无需硬件的界面自检命令：

```bash
cd /root/lerobot_project
. /usr/local/Ascend/ascend-toolkit/set_env.sh
/opt/smolvla_npu_test/bin/python 14_smolvla_atlas_live.py \
  --dashboard-self-test --dashboard-hold 60
```

网页、JPEG 仪表盘渲染和 `SAFE STOP` 联动均已通过自检。首次真实硬件复测时，从臂 6 号夹爪舵机因串联线断开而未响应，程序在连接阶段安全退出。重新接线后，六个舵机和两路真实相机全部上线，真实 `observe` 仪表盘测试通过；完整 NPU 推理耗时 `1119.9 ms`，日志为 `OBSERVE_ONLY_PASS`，没有发送动作，结束后串口、相机和 NPU 均正常释放。

### 18.7 与 PC `inference_with_live_monitor.py` 对齐的同步 rollout

早期 Atlas rollout 使用 `30 Hz / 10 步去噪 / 每 30 动作重规划 / 主体关节 2° 限幅`。60 秒遥测表明程序虽然发送了 788 条命令，但模型动作推进速度超过舵机跟随速度，安全目标反复形成锯齿，任务看起来停留在初始区域。

现已按 PC 端 LeRobot `SyncInferenceEngine + BaseStrategy` 的实际语义修改：

```text
控制频率：10 Hz
流匹配去噪：5 步
动作队列：完整执行 50 个动作后才重新观察和推理
主体关节相对硬限幅：10°/次
夹爪相对硬限幅：20°/次
结束行为：3 秒平滑返回启动姿态
```

PC 原程序的 `max_relative_target` 实际为 `None`，即完全不做机械臂相对目标硬限幅。Atlas 保留了 10°/20° 的最后一道保护，同时已经足以让轨迹明显推进。5 秒真实测试完整发送 50 个动作，5 步 NPU 推理约 `918 ms`，肘关节安全目标从约 `94°` 连续推进到约 `21°`，随后日志确认 `RETURN_TO_INITIAL_POSITION_PASS`。

现在原有命令无需额外参数即可使用 PC 同步逻辑：

```bash
/opt/smolvla_npu_test/bin/python 14_smolvla_atlas_live.py \
  --mode rollout \
  --motion-key ENABLE_ATLAS_MOTION \
  --duration 60
```

如需显式覆盖参数：

```text
--fps 10
--num-steps 5
--chunk-actions 50
--max-relative-target 10
--return-to-initial
```

`--max-relative-target 0` 可完全关闭限幅并与 PC 行为一致，但不建议在首次完整任务测试中使用。
