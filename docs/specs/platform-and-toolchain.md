# 平台与开发工具链方案（v3）：三环境分离

> **本版为何重写**：v2 建立在「裸机 + RTX 5090（Blackwell `sm_120`，32 GB）」之上，其中**整整一章（§5 Blackwell 深水区 + spconv 自编译）是最大风险点**。经实测澄清，目标机实为**公司 A6000（Ampere `sm_86`，48 GB）**，本机是 **GTX 1080（Pascal `sm_61`，8 GB）**。架构一变，v2 的结论**大半作废**：spconv 在 `sm_86` 上有预编译 wheel，整个「自编译」章节可以删掉。
>
> **依赖依据**：`docs/specs/algorithm-framework-and-selection.md`（P0—P8 选型）+ `.dsb/plans/v1-minimal-loop.md`（V1 八步流水线）。
> **本版原则**：**文档里出现的每个版本号，都必须是本机实测或官方发布页核对过的**，不用推测值。

---

## 〇、三环境分离（本方案的核心结构）

项目需要**三个不同环境**，它们的硬件与用途**本质不同**，混为一谈是 v1/v2 方案出错的总根源。

| # | 环境 | 硬件 | 用途 | 需要 GPU 训练？ |
|---|---|---|---|---|
| **E1** | **本机开发环境** | Windows 10 · GTX 1080（`sm_61`, 8 GB） | 写代码、跑 V1 纯几何链路、单元测试、可视化 | ❌ 不需要（1080 只够推理/小实验） |
| **E2** | **训练环境** | A6000（`sm_86`, 48 GB）· Linux | V2/V3 的 DL 训练（PTv3、ShapeVAE 等），**唯一需要训练的环境** | ✅ 需要 |
| **E3** | **文档/仓库** | 任意 | 本仓库文档、计划、spec | ❌ |

> **为什么必须分**：E1 与 E2 的技术栈**必然分叉**——E1 是 Windows + `cu121`（Pascal 上限），E2 是 Linux + 更现代 CUDA（`sm_86` 覆盖充分）。强行统一会两头不讨好：本机装不了新 torch（见 §2.2），服务器也不该被本机的旧版本拖住。

---

## 一、E1 本机开发环境（已搭建并验证 ✅）

### 1.1 实测基线

| 项 | 实测值 | 来源 |
|---|---|---|
| OS | Windows 10 (19045) | `systeminfo` |
| CPU / 内存 | 12 核 / 15.9 GB | 实测 |
| GPU | **NVIDIA GeForce GTX 1080** | `nvidia-smi` |
| Compute Capability | **`sm_61`（Pascal，2016）** | `torch.cuda.get_device_capability()` |
| 显存 | 8 GB | `nvidia-smi` |
| 驱动 | **560.94**（封装 CUDA 12.6） | `nvidia-smi` |
| Conda | `E:\AnacondaNew`（conda 23.7.4） | `conda --version` |
| Python | **3.11.16**（环境 `p2b`） | 实测 |
| 环境路径 | `E:\AnacondaNew\envs\p2b` | 实测 |

### 1.2 环境 `p2b` 的实际内容（全部实测通过）

| 组件 | 版本 | 备注 |
|---|---|---|
| numpy | **2.4.6** | ⚠️ 见 §1.4「与 v2 计划的一处差异」 |
| scipy | 1.17.1 | |
| scikit-learn | 1.9.1 | |
| pandas / matplotlib | 3.0.5 / 3.11.2 | |
| **numba** | 0.67.0 | JIT 实测通过 |
| **open3d** | **0.20.0** | 2000 点法向 + 体素下采样实测通过 |
| **cadquery** | **2.8.0**（自带 `cadquery-ocp` **7.9.3.1.1**） | 布尔 + 孔 + 倒角，`isValid=True`，`χ=2` |
| vtk | 9.6.2 | 随 open3d 装 |
| **torch** | **2.5.1+cu121** | CUDA build 12.1，cuDNN 90100 |
| **torchvision** | 0.20.1+cu121 | |
| sympy | **1.13.1** | 锁死：torch 2.5.1 要求 `sympy==1.13.1` |

**实测验证结论（真实功能测试，非只打印版本号）**：

```
[OK] numeric       numpy 2.4.6 / scipy 1.17.1 / sklearn 1.9.1 / pandas 3.0.5 / mpl 3.11.2
[OK] numba         numba JIT ok, sum=332833500
[OK] open3d        open3d 0.20.0, 2000 pts -> normals + voxel -> 620 pts
[OK] plane-fit     0.5mm-noise plane fit, normal error = 0.0212 deg
[OK] cadquery/occt cadquery 2.8.0, valid, V=18 E=27 F=11 chi=2 vol=7417.3 mm3
[OK] step-roundtrip STEP write+read ok, 20166 bytes
[OK] torch         torch 2.5.1+cu121 (cuda build 12.1, cudnn 90100)
[OK] torch-gpu     NVIDIA GeForce GTX 1080 sm_61 | matmul=61231 | backward ok
[OK] torchvision   torchvision 0.20.1+cu121, transforms ok
ALL CHECKS PASSED
```

> `plane-fit` 那一项是 V1 的核心算子：**0.5 mm 噪声下法向误差 0.0212°**，正值验证了「演绎优于插值」（对应 `min_feature_scale.py` 的判据 A）。
> `step-roundtrip` 证明 OCCT 的 STEP 写出+读回闭环可用，S8 导出路径无阻碍。

### 1.3 复现方式（已固化为三件交付物）

| 文件 | 作用 |
|---|---|
| `envs/setup-p2b.ps1` | 一键搭建脚本（conda 建环境 → 镜像装几何栈 → 下载 wheel 装 torch → 验证） |
| `envs/requirements-p2b.lock.txt` | 115 个包的完整锁定（torch 已改写为可复现的版本号形式，非本地路径） |
| `envs/verify_env.py` | 环境验证脚本（9 项真实功能测试，退出码 0 = 全通过） |

```bat
:: 一键复现
powershell -ExecutionPolicy Bypass -File envs\setup-p2b.ps1
:: 只验证
conda run --no-capture-output -n p2b python envs\verify_env.py
```

> `setup-p2b.ps1` **刻意只用 ASCII**：Windows PowerShell 5.1 在无 BOM 时按 GBK 读 `.ps1`，写中文会导致解析崩溃（本机实测踩过）。

### 1.4 下载通道（本机实测，重要）

| 通道 | 实测速度 | 结论 |
|---|---|---|
| **清华 PyPI 镜像**（`pypi.tuna.tsinghua.edu.cn`） | 快（几十 MB/s） | ✅ 装所有纯 Python / 通用 wheel |
| **SJTU PyTorch 镜像**（`mirror.sjtu.edu.cn/pytorch-wheels`） | **38—70 MB/s** | ✅ 下 2.4 GB 的 torch wheel 只花 33 秒 |
| 官方 `download.pytorch.org`（走 VPN 代理） | **0.08 MB/s** | ❌ 几乎卡死，不下 |
| 官方 `download.pytorch.org`（直连，不走代理） | 0.47 MB/s | ⚠️ 能用但慢（86 分钟） |

> **落地方案**：torch 系列一律**先从 SJTU 镜像下载 wheel 到 `E:\pip-cache\wheels`，再本地 `--no-index` 安装**。好处有三：**快**（33 秒 vs 86 分钟）、**可复现**（锁定本地文件）、**可离线重跑**。
>
> **代理真相**（实测）：本机 VPN（Lantern）真实端口是 **`127.0.0.1:51370`**；`pip.ini` 里写的 `50121` 是**废弃端口，无监听**。因此 `setup-p2b.ps1` **不把代理写死进配置**，只在需要时临时使用。
>
> **与 v2 计划的一处差异**：v2 建议 `numpy==1.26.4`（理由：numba 对新 numpy 敏感）。**实测 `numpy 2.4.6 + numba 0.67.0` 完全正常**（JIT 已验），故本版采用实际能跑通的组合，不照搬旧约束。

---

## 二、E1 的硬性天花板：Pascal 版本墙

### 2.1 事实：PyTorch 2.8 起移除了 Pascal 支持

| 来源 | 内容 |
|---|---|
| PyTorch 官方论坛（dev-discuss） | 「**Maxwell and Pascal architecture support removed in CUDA 12.8 and 12.9 builds**」 |
| pytorch/pytorch #157517 | `[release 2.8-2.9] Delete support for Maxwell, Pascal` |
| 用户实测报告（GTX 1050 `sm_61`） | PyTorch 2.8（cu128 wheel）「**只支持 sm_70 及以上**」，降级到 `2.4.1+cu121` 才恢复 |

**本机实测佐证**：

```
torch 2.5.1+cu121
arch_list = ['sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
含 sm_61 = True        ← 本机能用
GPU matmul OK          ← 实际跑通，非仅可导入
```

### 2.2 结论：本机 torch 上限 = `2.5.1+cu121`

- **`cu121` 是最后一个覆盖 `sm_61` 的 CUDA 分支**（`cu124` 只到 `sm_80+`，`cu126/cu128` 移除 Pascal）。
- **`torch 2.5.1` 是最后一个带 Pascal 内核的版本**（2.6/2.7 的 cu126 分支已不含 `sm_61`；2.8+ 明确移除）。
- 这不是「保守选择」，而是**物理上限**——本机想吃新 torch 已经不可能。

> **对项目的意义**：E1 的 torch 只用于**推理/小实验/接口联调**，**不承担训练**。真正的训练在 E2。因此「本机 torch 版本旧」**不影响项目**，只要 API 与 E2 兼容（避免用到 2.6+ 新 API）。

### 2.3 `sm_61` 与 `sm_86` 的巧合并存

`torch 2.5.1+cu121` 的 `arch_list` **同时含 `sm_61` 与 `sm_86`**。这意味着：

- E1（`sm_61`）与 E2（`sm_86`）**可以共用同一份 torch 版本**（`2.5.1+cu121` 是两边的安全交集）；
- 若 E2 想用更新的 torch（如 `2.7+cu126`），也**可以**——只需 E1 侧保持 `2.5.1`，两侧**接口/模型代码**保持一致即可（纯 Python 代码跨版本无碍）。

---

## 三、E2 训练环境（A6000，待搭建）

### 3.1 实测/核对过的事实

| 项 | 值 | 说明 |
|---|---|---|
| GPU | **NVIDIA A6000** | 用户已确认 |
| 架构 | **Ampere / `sm_86`** | Ampere 世代 |
| 显存 | **48 GB GDDR6** | 远超原文档假设的 32 GB |
| 原文档假设 | RTX 5090 / `sm_120` / 32 GB | ❌ **作废** |

### 3.2 v2「最大风险章节」在 A6000 上不成立（重要）

v2 §5 用整章论证「**spconv 需针对 `sm_120` 源码编译**」是最大风险点。**在 A6000（`sm_86`）上此问题消失**——实测 PyPI 上有现成 wheel：

| 包 | 版本 | Linux wheel（`cp311`） | Windows wheel |
|---|---|---|---|
| `spconv-cu126` | **2.3.8** | ✅ `manylinux_2_28_x86_64` | ✅ `win_amd64` |
| `cumm-cu126` | **0.8.2** | ✅ `manylinux_2_28_x86_64` | ✅ `win_amd64` |
| `spconv-cu120` | 2.3.6 | ✅ `manylinux_2_17_x86_64` | ✅ `win_amd64` |

> **⚠️ 更正**：v2 附录曾断言「spconv 仅 manylinux，无 win 轮子」——**实测为假**，`spconv-cu120/126` **确有 `win_amd64` 轮子**。但它**在本机无意义**（1080 训练不行，V1 也不需要），所以本机不装。
>
> **结论**：E2 上 `pip install spconv-cu126` 即可，**省掉 v2 整章的自编译工程量**（`TORCH_CUDA_ARCH_LIST`、`nvcc`、`cumm` 版本配套全部不需要）。

### 3.3 E2 建议的版本矩阵（`sm_86` 覆盖充分，可自由选）

| 组件 | 建议 | 理由 |
|---|---|---|
| OS | Ubuntu 22.04 / 24.04 | 公司服务器常见基线 |
| 驱动 | ≥ 550 | 支持 CUDA 12.6 |
| CUDA Toolkit | **12.6**（或 12.4） | `sm_86` 有预编译 wheel 的**最新**分支 |
| Python | 3.11 | 与 E1 一致，wheel 最全 |
| **PyTorch** | **2.7.x + `cu126`**（或更高） | `sm_86` 在新版中持续被支持；比 E1 新 |
| **spconv** | **`pip install spconv-cu126`** | 预编译 wheel，**无需自编译** |
| 环境管理 | conda / mamba | 与 E1 一致 |
| **训练框架** | Lightning + Hydra + MLflow/W&B | 结构化 trainer + 配置 + 追踪 |
| 数据格式 | HDF5 / WebDataset | 大点云分片 |

> **纪律**：E2 具体版本**以机器实际可用为准**（公司机器可能已有既定镜像/驱动），落地时用 `nvidia-smi` + `nvcc -V` 核对后再定，**不照抄本表**。

### 3.4 E1 ↔ E2 协作方式

| 事项 | 方式 |
|---|---|
| 分工 | E1 写代码 + 跑 V1；E2 只做训练 |
| 代码同步 | Git（本仓库），E2 `git clone` 拉取 |
| 数据同步 | V1 数据小（合成，MB 级）走 Git LFS；训练集（GB 级）走内网/对象存储 |
| 依赖一致 | **`envs/` 下的锁定文件是 E1 的**；E2 另写 `envs/requirements-train.lock.txt` |
| 训练产物 | checkpoint 不入库（`.gitignore` 已排除 `*.pt`/`*.ckpt`） |

---

## 四、三阶段工具链（跨环境对照）

### 4.1 V1（纯几何，**E1 本机即可跑通**）

| 步 | 模块 | 首选（已验证） |
|---|---|---|
| S1 | 预处理 | **Open3D 0.20.0**（体素/离群/kNN-PCA 法向） |
| S2 | 分片+类型 | 自研启发式（V1 人工/规则；V2 换 DL） |
| S3 | 拟合 | **NumPy/SciPy**（`least_squares` + 自研 RANSAC） |
| S4 | 残差诊断 | SciPy `stats` + 鲁棒核（Huber/Tukey/95 分位） |
| S5 | MDL | **自研** |
| S6 | 全局正则 | **SciPy `least_squares`**（Ceres 备选，非必需） |
| S7 | 拓扑构造 | **OCCT**（经 `cadquery-ocp` 7.9.3.1.1 提供） |
| S8 | 验证+导出 | **OCCT**（`BRepCheck` / STEP AP242），实测往返通过 |
| 可视化 | | Open3D 窗口 / matplotlib / VTK |

> **V1 全部依赖已在 E1 实测通过**，不需要 GPU，不需要 E2。这是「先搭环境再谈 CAD 真值」的直接兑现。

### 4.2 V2（学习式感知，**需要 E2**）

| 阶段 | 首选 | 显存（A6000 48 GB 下充裕） |
|---|---|---|
| P1 逐点分割/类型 | **PTv3（Pointcept）** | 8—16 GB |
| P1 稀疏卷积 | **spconv-cu126（预编译）** | 同上 |
| P3 边界性质头 | PTv3 共享骨干 + 多头 | 同上 |
| P5 latent 几何 | ShapeVAE 冻结 encoder + 探针 | ~16 GB |

### 4.3 V3（装配级）

GNN（PyG）/ 因子图（Ceres/GTSAM）/ 本地 LLM 提议 / ROS 2 主动消歧 / FastAPI 服务化。**均在 E2 上进行**。

---

## 五、显存预算（A6000 48 GB）

| 模型/任务 | 估算显存 | 48 GB 判断 |
|---|---|---|
| PTv3 分割（批 16，BF16） | 16—24 GB | ✅ 充裕 |
| ShapeVAE | 12—20 GB | ✅ |
| GNN 装配关系 | 2—6 GB | ✅ |
| SDF/扩散补全 | 16—24 GB | ✅ |
| 本地 LLM 32B-4bit 推理 | 18—20 GB | ✅ |

> 本框架自估**可训练参数 < 100 M**（《框架》10.1）→ **A6000 单卡对整个 V1—V3 都是充裕的**，瓶颈在数据吞吐，不在显存。

---

## 六、待办（E2 落地时补）

| # | 事项 | 状态 |
|---|---|---|
| 1 | 拿到 A6000 机器：OS / 驱动 / CUDA / `sudo` / 磁盘 / SSH | ⏳ 待用户提供 |
| 2 | 按 §3.3 核对版本 → 写 `envs/setup-train.ps1`（或 `.sh`） | ⏳ |
| 3 | 生成 `envs/requirements-train.lock.txt` | ⏳ |
| 4 | 数据同步方案（Git LFS vs 对象存储） | ⏳ |

---

## 七、与旧版一致、不因换机而改的部分

- **算法选型不变**：P0—P8 与首选算法全部保留（见 `algorithm-framework-and-selection.md`）。
- **核心原则不变**：模型只输出**决策**，几何参数由**解析拟合**算出；传统几何是精度来源；拓扑先验（欧拉 `V−E+F−L=2(S−G)`）强制执行。
- **符号端不变**：OCCT（B-rep / STEP AP242）。
- **目录结构不变**：`src/{io,segment,fit,diagnose,mdl,regopt,topology,verify,export,eval}`。

---

## 八、一句话总结

> 项目是**三环境**：**E1 本机（Windows + GTX 1080 `sm_61`）只做开发与 V1 纯几何链路，已搭好并实测全通过**；**E2（公司 A6000 `sm_86` 48 GB）只做训练，spconv 有预编译 wheel，v2 的自编译风险章节作废**；E3 是文档。本机 torch 被 Pascal 版本墙锁死在 **`2.5.1+cu121`**（PyTorch 2.8 起移除 Pascal），但这不影响项目，因为训练不在本机。**下载统一走镜像**（清华 PyPI + SJTU PyTorch，实测 38—70 MB/s），不依赖 VPN。
