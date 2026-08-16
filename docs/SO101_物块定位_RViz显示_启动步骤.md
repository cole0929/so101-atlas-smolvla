# 黄色物块定位 → RViz 显示（纯净启动步骤）

> 适用：前置相机定位黄色物块，RViz 实时显示位置（3D 小球 + 坐标文字）
> 更新：2026-08-16

## 架构

```text
板端 19_locate_object.py（相机检测 → base_link 3D 坐标）
        │ 写文件 /root/located_object.json
        ▼
Windows poll_located_object.ps1（scp 拉取）
        │ 写到 F:\robot_arm_atlas\.tmp\located_object.json
        ▼
WSL located_object_bridge.py（读 /mnt/f 文件 → 发布 Marker）
        │ /located_object
        ▼
RViz（Located Object 显示层）
```

## 启动步骤（按顺序）

### 第 1 步：开发板 — 只读桥（提供机械臂模型数据）

**MobaXterm**（Ascend-devkit 会话）：

```bash
systemctl start so101-rviz-bridge.service
systemctl is-active so101-rviz-bridge.service
# 预期：active
```

### 第 2 步：Windows — 文件拉取循环（保持运行）

**Windows PowerShell**（新窗口）：

```powershell
powershell -ExecutionPolicy Bypass -File F:\robot_arm_atlas\ros2\poll_located_object.ps1
```

> 预期：绿色提示 `Polling root@192.168.0.2:/root/located_object.json ...`

### 第 3 步：开发板 — 物体定位（保持运行）

**MobaXterm**（新标签页，Duplicate）：

```bash
cd /root/lerobot_project
/opt/lerobot061/bin/python 19_locate_object.py --file /root/located_object.json
```

> 预期：每 0.5s 打印一行 `obj: (xx.x, xx.x, 0.0) cm`（黄色物块位置）

### 第 4 步：WSL — 修复 ros2 daemon（一次性）

**Windows PowerShell**：

```powershell
wsl -d Ubuntu-22.04
```

```bash
ros2 daemon stop
exit
```

> 原因：WSL 的 ros2 daemon 与 Zenoh 不兼容，会导致 CLI 查询失败。停掉后节点直连正常。

### 第 5 步：Windows — 启动 RViz + 全部图层

**Windows PowerShell**（新窗口，保持打开）：

```powershell
powershell -ExecutionPolicy Bypass -File F:\robot_arm_atlas\ros2\start_so101_trajectory_vcxsrv.ps1
```

> 自动启动：retime + robot_state_publisher + 轨迹 recorder + 数字孪生图层 + object bridge + RViz2

## 预期结果

RViz 中可见：

| 图层 | 内容 |
|---|---|
| 机械臂模型 | 跟随真实从臂（只读桥数据） |
| 黄色小球 + 坐标文字 | 物块 3D 位置（`/located_object`） |
| 绿色工作区 + 红色禁区墙 | 实际工作空间边界 |
| 关节角度标注 | 六关节实时角度 |

## 停止

| 终端 | 操作 |
|---|---|
| MobaXterm 定位终端 | Ctrl+C |
| PowerShell 拉取循环 | Ctrl+C |
| PowerShell RViz 窗口 | Ctrl+C |
| MobaXterm | `systemctl stop so101-rviz-bridge.service`（可选） |

## 常见问题

| 现象 | 处理 |
|---|---|
| RViz 无黄色小球 | 左侧 Displays 面板勾选「Located Object」；确认终端 2/3 在跑 |
| 定位脚本报 `camera read failed` | 相机被占用，检查是否有其他进程开相机 |
| RViz 无机械臂模型 | 第 1 步只读桥 active；`ros2 daemon stop` 已执行 |
| 物块位置不准 | 确认黄色物块 HSV 阈值（默认 20-35 色相）；工作台高度 z=0 假设 |
