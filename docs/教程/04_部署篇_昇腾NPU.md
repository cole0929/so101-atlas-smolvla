# 模块 4：部署篇 —— 昇腾 NPU

> 学完本篇你能回答：模型怎么从「服务器上能跑」变成「开发板上能跑」？ONNX/OM/ATC/pyACL 都是什么？

---

## 1. 本篇目标

- 理解为什么模型不能直接搬到板子上（PyTorch 跑不动、格式不对）
- 理解 ONNX → OM → pyACL 这条部署链路上每一步在干什么
- 理解为什么要拆成 4 个子图分别转换
- 能说出 NPU 推理的全过程（vision → prefix → denoise 串联）
- 理解 golden case 对比验证的重要性

## 2. 一句话版本

> PyTorch 模型不能直接上 NPU。我们把 SmolVLA 拆成 4 个子图
> （vision / state_proj / prefix / denoise），用华为的 ATC 工具
> 把每个 ONNX 文件编译成 NPU 专属的 OM 文件，再用 pyACL
> （Ascend 计算库的 Python 接口）按顺序调用这 4 个 OM，
> 一次推理约 918ms，驱动机械臂执行 5 秒动作。

## 3. 背景故事：为什么要这么折腾

### 3.1 问题：模型跑在服务器上 ≠ 跑在板子上

训练好的 SmolVLA 是 PyTorch 格式（在 A100 上训练）。但：

- 板子（aarch64）上跑 PyTorch 的 CPU 推理要 **73 秒/帧**——完全没法控制机械臂
- NPU 不认识 PyTorch 模型，只认自己的格式（OM）
- 板子内存有限，整个模型搬上去也放不下（所以拆子图、逐个优化）

### 3.2 解决：一条「翻译 + 编译」流水线

```text
PyTorch 模型（训练产物）
   │ 导出（torch.onnx.export）
   ▼
ONNX（通用交换格式，模型界的 JPEG）
   │ ATC 编译（华为模型转换工具，带算子融合/图优化）
   ▼
OM（NPU 专属可执行文件）
   │ pyACL 加载调用（Ascend 计算库的 Python 接口）
   ▼
NPU 推理 → 输出 50×6 动作块
```

> 类比：ONNX 像「通用文稿」，ATC 是「NPU 专属编译器」，
> 把它编译成板子 CPU/NPU 直接能跑的机器码（OM），
> pyACL 是「运行时驱动」，负责把数据喂进去、把结果拿出来。

### 3.3 为什么要拆 4 个子图

整个模型一起转 ONNX/OM 会失败（算子不支持、内存爆）。按功能拆开：

| 子图 | 干什么 | 部署难点 |
|---|---|---|
| `vision.om` | 图像编码 | Transformers 动态 mask 无法导出 → 固定 512×512 |
| `state_proj.om` | 状态投影 | 输入必须 6→32 补零（漏了会报错） |
| `prefix_origin.om` | 语言前缀预填充 | FP16 深层误差累积 → 高精度模式 |
| `denoise.om` | 动作去噪 | 5 步去噪 + Euler 积分 |

拆开后每个子图单独转换、单独验证（对应 `npu_tests/` 里 01~13 号脚本的分步验证思路），最后在运行时串起来。

## 4. 核心概念（大白话 + 面试版）

| 概念 | 大白话 | 面试怎么讲 |
|---|---|---|
| **ONNX** | 模型通用交换格式 | 跨框架标准，任何框架可导出/读取 |
| **OM** | 华为 NPU 可执行模型文件 | NPU 专属编译产物 |
| **ATC** | 华为模型转换工具 | 把 ONNX 编译成 NPU 机器码的编译器 |
| **pyACL** | Ascend 计算库的 Python 接口 | NPU 的 Python 驱动 |
| **AclLite** | 华为对 pyACL 的二次封装 | 更省事的 NPU 工具箱 |
| **CANN** | 华为 AI 计算框架（类似 CUDA） | 昇腾平台软件栈（本项目 7.0.RC1） |
| **KV cache** | 缓存已算过的注意力结果 | 记住看过的内容不重复算，推理提速关键 |
| **golden case** | 标准输入输出样本，用于对比验证 | 用 CPU 参考结果验证 NPU 输出一致 |
| **310B1** | 板子上 NPU 的型号 | Ascend 310B1 推理芯片 |

## 5. 实际操作：部署链路怎么验证

工作区 `npu_tests/` 里就是完整的分步验证链路（01~13）：

```bash
# 每个脚本做什么（编号顺序 = 验证顺序）
01  build_matmul_onnx.py       # 先验证 NPU 能不能算矩阵乘法（冒烟测试）
02  run_matmul_om.py           # 跑转好的 matmul OM
03  export_smolvla_state_proj.py   # 导出 state_proj 子图
04  run_smolvla_state_proj_om.py   # NPU 跑 state_proj
05  export_smolvla_vision.py       # 导出视觉编码器
06  run_smolvla_vision_om.py
07  compare_vision_original_cpu.py # CPU vs NPU 对比
08  export_smolvla_prefix.py       # 导出语言前缀
09  run_smolvla_prefix_om.py
10  export_smolvla_denoise.py      # 导出动作去噪器
11  run_smolvla_denoise_om.py
12  validate_denoise_wrapper.py    # 验证封装正确性
13  run_smolvla_full_npu.py        # 全模型 NPU 推理（串联 4 个 OM）
```

**核心方法论：一步步来，每步都有 golden case 对比。**
先跑通最简单的 matmul，确认「NPU 能用」；再逐个子图导出→转换→对比；
最后才串联全流程。这就是为什么这个项目部署能稳定跑通——**分而治之 + 全程验证**。

## 6. 踩过的坑（真实记录，部署环节最多）

| 坑 | 解决 |
|---|---|
| ATC 绑定 Python 3.10，与项目 3.12 冲突 | 建 `/opt/cann_py310_deps` 用 PYTHONPATH 注入 |
| 板端缺 transformers/accelerate 等依赖 | 补离线 wheel，独立环境不污染采集环境 |
| Transformers 动态视觉 mask 无法导出 ONNX | 固定 512×512、固定 position id、eager attention |
| Prefix FP16 深层误差累积 | `must_keep_origin_dtype` + `high_precision` |
| BF16 权重装进 FP32 骨架 | `load_state_dict(..., assign=True)` |
| state OM 报「输入 24 bytes / 需 128 bytes」 | 漏了 6→32 补零 |
| 相机映射写死节点号 | 按 USB PID 动态识别 |

## 7. 面试怎么讲（1 分钟版）

> 部署是项目里技术难点最集中的环节。SmolVLA 是 PyTorch 模型，
> 而开发板是 aarch64 + 昇腾 NPU，不能直接运行。
> 我们的方案是：先把模型拆成 vision、state_proj、prefix、denoise
> 四个子图导出为 ONNX——拆分是为了规避整模型导出失败，
> 也便于逐个验证；然后用华为 ATC 工具编译成 OM 格式；
> 运行时用 pyACL + AclLite 封装，按 vision → prefix（KV prefill）→
> denoise×5 的顺序串联推理，实测单次完整推理约 918ms。
> 整个迁移过程我采用了「分而治之 + golden case 对比」的验证方法：
> 从最简单的矩阵乘法开始，逐步验证每个子图的 CPU/NPU 输出一致性，
> 最后才做全链路串联，保证部署正确性可追溯。

## 8. 自查清单

- [ ] 能画出 ONNX → OM → pyACL 的部署链路图
- [ ] 能说出 ATC、OM、pyACL 各是什么
- [ ] 能说出为什么拆 4 个子图、分别是哪 4 个
- [ ] 能说出 golden case 验证的思路（对比 CPU/NPU 输出）
- [ ] 能说出一次推理约多少毫秒（918ms）
- [ ] 能说出 KV cache 为什么重要（推理提速）

---
下一站：[模块 5：孪生篇 —— ROS2 数字孪生](05_孪生篇_ROS2数字孪生.md)
