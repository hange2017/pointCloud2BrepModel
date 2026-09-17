# 环境搭建 spec（三环境 · 操作手册）

> **与 `docs/specs/platform-and-toolchain.md` 的分工**
> - `docs/specs/platform-and-toolchain.md` = **平台策略**（为什么这么选、架构天花板、选型矩阵）
> - 本文 = **操作手册**（怎么装、怎么验证、怎么复现，含确切命令与实测数字）
>
> 状态：**E1 已完成并实测通过**；E2 待机器就绪。

---

## 一、环境清单

| 环境 | 硬件 | 用途 | 状态 |
|---|---|---|---|
| **E1 本机开发** | Windows 10 (19045) · GTX 1080（`sm_61`, 8 GB）· 12 核 / 15.9 GB | 写代码、V1 纯几何链路、单测、可视化 | ✅ **已通过** |
| **E2 训练** | 公司 A6000（`sm_86`, 48 GB）· Linux | V2/V3 的 DL 训练 | ⏳ 待机器就绪 |
| **E3 文档** | 任意 | 本仓库文档 | — |

**E1 环境标识**：conda 环境名 `p2b`，路径 `E:\AnacondaNew\envs\p2b`，Python **3.11.16**。

> 装到 **E 盘**（C 盘空间紧张，用户明确要求）。conda 根目录 `E:\AnacondaNew`。

---

## 二、E1 交付物（三个文件，均已入库）

| 文件 | 作用 |
|---|---|
| `envs/setup-p2b.ps1` | 一键搭建：建 conda 环境 → 清华镜像装几何栈 → SJTU 镜像下 wheel 装 torch → 调验证脚本 |
| `envs/requirements-p2b.lock.txt` | 完整锁定（115 包）。torch 已从本地 `file://` 路径改写为 `torch==2.5.1+cu121` 形式 |
| `envs/verify_env.py` | 9 项**真实功能测试**（不是打印版本号），退出码 0 = 全通过 |

```bat
:: 一键复现（幂等：环境已存在则复用）
powershell -ExecutionPolicy Bypass -File envs\setup-p2b.ps1

:: 只跑验证
conda run --no-capture-output -n p2b python envs\verify_env.py

:: 只要纯几何栈、不要 torch
powershell -ExecutionPolicy Bypass -File envs\setup-p2b.ps1 -SkipTorch
```

### 2.1 为什么用 PowerShell 而不是 .bat

**实测坑**：本机 `cmd.exe` 传参会把带空格/中文的引号参数拆坏（本项目提交信息就踩过）。而 `.ps1` 又有一个反向坑——**PowerShell 5.1 在无 BOM 时按 GBK 读文件**，写中文会解析崩溃。因此：

> **`envs/setup-p2b.ps1` 刻意只用 ASCII**，并在文件头注明原因。改这个文件时**不要加中文**。

---

## 三、E1 实测结果（复现基线）

`verify_env.py` 的实际输出（2026-09 实测）：

```
python 3.11.16 @ E:\AnacondaNew\envs\p2b\python.exe
======================================================================
[OK] numeric       numpy 2.4.6 / scipy 1.17.1 / sklearn 1.9.1 / pandas 3.0.5 / mpl 3.11.2
[OK] numba         numba JIT ok, sum=332833500
[OK] open3d        open3d 0.20.0, 2000 pts -> normals + voxel -> 620 pts
[OK] plane-fit     0.5mm-noise plane fit, normal error = 0.0212 deg
[OK] cadquery/occt cadquery 2.8.0, valid, V=18 E=27 F=11 chi=2 vol=7417.3 mm3
[OK] step-roundtrip STEP write+read ok, 20166 bytes
[OK] torch         torch 2.5.1+cu121 (cuda build 12.1, cudnn 90100)
[OK] torch-gpu     NVIDIA GeForce GTX 1080 sm_61 | matmul=61231 | backward ok
[OK] torchvision   torchvision 0.20.1+cu121, transforms ok
======================================================================
ALL CHECKS PASSED
```

**九项分别证明了什么**（V1 依赖逐条兑现）：

| 检查 | 兑现的能力 |
|---|---|
| numeric | S3/S4/S5/S6 的数值与优化底座 |
| numba | 热点循环 JIT（RANSAC 迭代） |
| open3d | S1 预处理：下采样 + 法向估计 |
| plane-fit | **V1 核心算子**：0.5 mm 噪声下法向误差 0.02° →「演绎优于插值」 |
| cadquery/occt | S7 拓扑构造：布尔/倒角有效，`χ=2`（闭合实体） |
| step-roundtrip | **S8 导出闭环**：STEP 写出+读回无损 |
| torch / torch-gpu / torchvision | 推理与接口联调（**非训练**） |

---

## 四、E1 硬约束（不可违背）

| 约束 | 值 | 原因 |
|---|---|---|
| torch 上限 | **`2.5.1+cu121`** | PyTorch 2.8 起移除 Pascal；`cu121` 是最后覆盖 `sm_61` 的分支 |
| sympy | **`==1.13.1`** | `torch 2.5.1` 硬性要求（实测装 1.14.0 会冲突告警） |
| 不装 spconv | — | 本机不训练、V1 不需要；且 1080 训练本就不行 |
| 下载通道 | 清华 PyPI + SJTU PyTorch | 官方源走 VPN 仅 0.08 MB/s，几乎卡死 |
| 代理 | **临时用，不写死** | 真实端口 `127.0.0.1:51370`（Lantern）；`pip.ini` 里的 `50121` 是废弃端口 |

> **为什么 torch 版本旧不影响项目**：E1 只做开发与推理，**训练在 E2**。只要 E1 代码不调用 2.6+ 的新 API 即可。

---

## 五、E2 落地清单（待机器就绪后执行）

| # | 步骤 | 命令 / 判据 |
|---|---|---|
| 1 | 核对机器 | `nvidia-smi`（型号/驱动/显存）、`nvcc -V`、`lsb_release -a`、`df -h` |
| 2 | 建环境 | `conda create -y -n p2b-train python=3.11` |
| 3 | 装 torch | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| 4 | 装稀疏卷积 | `pip install spconv-cu126` ← **预编译 wheel，无需自编译**（`sm_86` 有现成轮子） |
| 5 | 装训练框架 | Lightning + Hydra + MLflow（或 W&B） |
| 6 | 验证 | `python -c "import torch,spconv;print(torch.cuda.get_device_name(0),spconv.__version__)"` |
| 7 | 生成锁定 | `pip freeze > envs/requirements-train.lock.txt` |

**关键结论**：A6000 是 `sm_86`，`spconv-cu126 2.3.8` 有 `manylinux_2_28_x86_64` 轮子 → **原文档整章「spconv 自编译」作废**，省掉 `nvcc` / `TORCH_CUDA_ARCH_LIST` / `cumm` 版本配套全部工程量。

---

## 六、E1 ↔ E2 协作约定

| 事项 | 约定 |
|---|---|
| 代码同步 | Git 仓库；E2 `git clone` 拉取，不复用 E1 的 conda 环境 |
| 数据同步 | V1 合成数据（MB 级）→ Git LFS；训练集（GB 级）→ 内网 / 对象存储 |
| 依赖文件 | E1 = `requirements-p2b.lock.txt`；E2 = `requirements-train.lock.txt`（分开，**不混用**） |
| 训练产物 | checkpoint 不入库（`.gitignore` 已排除 `*.pt`/`*.pth`/`*.ckpt`） |
| 接口纪律 | 模型代码保持**纯 Python + 版本无关**，避免 E1/E2 因 torch 版本差而分叉 |

---

## 七、验收标准

**E1（已达）**：`verify_env.py` 退出码 0，9 项全 OK。
**E2（待）**：torch 可用 GPU + `import spconv` 成功 + 一个最小训练 step 能跑通（`loss.backward()`）。

---

## 八、附：本机曾出现的两个环境级问题（已解决，留档）

1. **`pip.ini` 配了失效代理** `127.0.0.1:50121`（端口无监听）→ 一度导致 pip 全部超时。**对策**：不把代理写进配置，需要时命令行 `--proxy` 临时指定，或 `--noproxy "*"` 绕过。
2. **`pip install` 走官方源极慢** → **对策**：几何栈走清华镜像，torch 走 SJTU 镜像**先下 wheel 再本地装**（实测 2.4 GB / 33 秒）。
