# Atlas + SO-ARM101 重力补偿与拖动保持任务

> 建立日期：2026-08-15

## 目标

用手推动 Follower 关节时自动进入低阻力状态，用户停稳后先将目标位置设为当前编码器位置，再恢复位置环保持，避免重新上力时跳变。

## 硬件边界

SO-ARM101 Follower 使用的 STS3215 支持位置、速度、PWM 开环和步进模式，但没有可下发的目标电流/目标力矩寄存器。因此第一版实现是“自动离合 + 位置保持”，不是工业协作臂的动力学力矩前馈。

## 实施顺序

1. 板端恢复 SSH 后，先运行只读 `monitor`，采集位置、速度、负载、电流和温度；
2. 支撑机械臂，只在 `wrist_flex` 上运行 10 秒自动离合；
3. 验证无上力跳变、无抖动、无过热，再逐个扩展到腕部、胘部和肩部；
4. 记录各关节合适的触发位移、静止速度和停稳时间；
5. 如果自动离合仍有明显下坠，再评估基于姿态的目标位置微偏置补偿，不直接修改舅机 EEPROM 保护参数。

## 程序

```text
/root/lerobot_project/09_hand_guiding_hold.py
```

只读命令（默认，不写舅机寄存器）：

```bash
cd /root/lerobot_project
/opt/lerobot061/bin/python 09_hand_guiding_hold.py \
  --mode monitor --motors wrist_flex --duration-s 10
```

首次单关节控制命令（必须人工支撑机械臂）：

```bash
cd /root/lerobot_project
/opt/lerobot061/bin/python 09_hand_guiding_hold.py \
  --mode auto --motors wrist_flex --duration-s 10 --enable-control
```

## 安全门槛

- 第一次不同时释放肩、胘关节；
- 不在手、脸、线缆或易碎物位于工作区时运行；
- 只有显式传入 `--enable-control` 才允许写入目标位置和扭矩开关；
- 温度达到 60°C 立即停止并关闭扭矩；
- `Ctrl+C` 默认先捕获当前位置并保持；仅在人已托住机械臂时使用 `--release-on-exit`；
- 任何突跳、异响、剧烈抖动或通信错误都立即切断执行器电源。

## 当前状态

- 控制原型已部署到 Atlas，板端 Python 语法检查通过；
- Follower 已按 USB 序列号正确识别为 `/dev/ttyACM1`；
- 5 秒单关节和 1 秒全关节只读监测通过，六个舅机温度为 42–44°C，通信无报错；
- 只读阶段没有向机械臂写入新参数，没有切换扭矩，没有执行动作；
- 下一安全门是现场人员托住机械臂后，执行 10 秒 `wrist_flex` 单关节自动离合。
