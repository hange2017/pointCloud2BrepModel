# 平台与开发工具链方案（v2）：非虚拟机 + RTX 5090 工作站

> 修订说明：v1 基于「VirtualBox 客户机 / 无 GPU / 6 GB 内存」的实测环境，结论是「训练必须外置」。**本版按用户目标环境重写：裸机（非虚拟机）+ RTX 5090 高显存显卡**，因此**训练、推理、传统几何全流程均可本地化**。
> 依赖依据：`docs/specs/algorithm-framework-and-selection.md`（P0—P8 选型）+ `.dsb/plans/v1-minimal-loop.md`（V1 八步流水线）。

---

## 〇、必须先澄清的两件事（影响全部版本选型）

### 0.1 RTX 5090 的显存与架构（请核对卡型号）

| 属性 | RTX 5090 | 说明 |
|---|---|---|
| 架构 / 代号 | **Blackwell / GB202** | 与 Ada、Ampere 是**不同的 CUDA 架构** |
| Compute Capability | **`sm_120`** | 编译 `TORCH_CUDA_ARCH_LIST` 必须含 `12.0` |
| 显存 | **32 GB GDDR7**（官方规格），512-bit | **不是 24 GB** |
| 带宽 | ~1.79 TB/s | 适合大点云吞吐 |

> **重要**：题述「24 G 显存」与 RTX 5090 官方规格（32 GB）不符。若你手头真是 24 GB，那更可能是 **RTX 4090（Ada，`sm_89`）或 3090（Ampere，`sm_86`）**。**这两条路的 CUDA / PyTorch / spconv 编译目标完全不同**（见 §2）。本方案按「Blackwell `sm_120` + ≥24 GB」双分支规划，落地前请先用 `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv` 确认型号与算力。

### 0.2 非虚拟机解除的约束（v1 → v2 对照）

| 约束项 | v1（VirtualBox） | v2（裸机 + 5090） | 影响 |
|---|---|---|---|
| GPU | 无（`nvidia-smi`/`nvcc` 缺） | **5090，可直连** | **训练本地化**（v1 必须外置） |
| 内存 | 6 GB（可用 1.9 G） | 建议 **≥32 GB**（64 G 更稳） | 可载百万级点云、可交互可视化 |
| OpenGL | llvmpipe 软渲染 | **原生驱动** | 可开 Open3D 交互窗口 / PyVista |
| OCCT | 无 conda，`pythonocc-core` 不在 PyPI | **有 conda/mamba** | 可用 conda 装 `pythonocc-core`，不必依赖 apt 7.5 |
| cmake | 3.19.6（偏旧） | 可任意升级 | 可源码编译（spconv 需要） |
| 训练 | 不可能 | **本地训练 PTv3/GNN** | 排期不再依赖外部 GPU 机 |

---

## 一、硬件与系统基线（Blackwell 专属）

| 项 | 目标配置 | 备注 |
|---|---|---|
| GPU | RTX 5090（`sm_120`，32 GB） | 或 ≥24 GB 的 Ada/Ampere（见 §0.1） |
| **NVIDIA 驱动** | **≥ 570 系列** | Blackwell 支持从 570 起步；Ubuntu 22.04 经 `graphics-drivers` PPA 或 CUDA apt 源安装 |
| **CUDA Toolkit** | **12.8+**（推荐 12.9 / 13.x 现行版） | **12.8 是首个支持 Blackwell（`sm_100`/`sm_120`）的版本**，低于此无法为 5090 编译 |
| cuDNN | 9.x（匹配 CUDA） | 随 notebook / apt 安装 |
| OS | Ubuntu 22.04 LTS（可考虑 24.04） | 22.04 内核 ≥ 6.8 对 5090 更稳；24.04 驱动支持更省事 |
| 内存 | **≥ 32 GB**（建议 64 GB） | 点云 + 数据集缓存 + 可视化并发 |
| 磁盘 | **NVMe ≥ 1 TB**（建议 2 TB） | 数据集/权重放本地 NVMe，避免网络盘 |
| Python | **3.11 / 3.12**（conda 管理） | 与 CUDA/PyTorch wheel 对齐；不再手工 venv |

> 驱动与 CUDA 是**两个独立版本**：装 PyTorch 的 cu128 wheel 时，**系统驱动必须 ≥ 570**；而编译 spconv 用的是 **CUDA Toolkit 12.8+ 的 nvcc**，二者不可混淆。

---

## 二、版本矩阵（决定成败，先钉死）

| 组件 | 推荐版本 | 为何是它 | 验证命令 |
|---|---|---|---|
| NVIDIA 驱动 | **570+**（当前 stable） | 5090 最低门槛 | `nvidia-smi`（看 Driver Version / CUDA Version） |
| CUDA Toolkit | **12.8 / 12.9** | 首个支持 `sm_120` | `nvcc -V` |
| **PyTorch** | **2.7.0+（`cu128` 轮子）** | **PyTorch 2.7 是首个官方支持 Blackwell 的稳定版**；2.8/2.9/2.10 继续 | `python -c "import torch;print(torch.__version__, torch.version.cuda, torch.backends.cuda.is_built())"` |
| Python | 3.11 或 3.12 | 与 torch wheel ABI 匹配 | `python -V` |
| 环境管理 | **mamba/micromamba**（或 uv） | CUDA 生态版本管理，解决 v1 无 conda 之痛 | `mamba --version` |
| **spconv** | **2.3.x，需针对 `sm_120` 自编译** | 预编译 wheel 无 Blackwell 目标（**最大坑**，见 §5） | `python -c "import spconv;print(spconv.__version__)"` |
| cuDNN | 9.x | 随 CUDA | `python -c "import torch;print(torch.backends.cudnn.version())"` |
| Open3D | 0.19 | py3.12 wheel 可用，GPU 版本可选 | `python -c "import open3d;print(open3d.__version__)"` |
| CadQuery / OCP | 2.8（自带 OCCT ~7.8） | OCCT 绑定 pip 路线 | `python -c "import OCP;print('ok')"` |
| pythonocc-core | conda 版（可选） | 有 conda 后可用官方绑定 | `conda list pythonocc-core` |
| Ceres / Eigen / CGAL | apt：Ceres 2.0、Eigen 3.4、CGAL 5.4 | C++ 性能核 | `pkg-config --modversion ceres-solver` |

---

## 三、环境管理：从 venv 升级到 mamba（非虚拟机才现实）

v1 因无 conda 只能 venv + apt OCCT。裸机可装 mamba，获得三样好处：**CUDA 版本并存管理、`pythonocc-core` 直接可用、SciPy/Open3D 等二进制依赖一致**。

```bash
# 装 micromamba（轻量，无需 root，避免 conda 慢）
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba create -y -n wr python=3.12
./bin/micromamba activate wr

# 环境导出即基础设施即代码（团队复现）
./bin/micromamba env export -n wr > environment.yml
```

---

## 四、三阶段工具链（全部本地化）

### 4.1 阶段一（V1）：传统几何栈 —— 与 v1 一致，本机 CPU/新机皆可

| V1 步 | 模块 | 首选 | 备选 |
|---|---|---|---|
| S1 | 预处理 | **Open3D 0.19**（体素下采样/统计离群/kNN-PCA 法向） | PCL 1.14、trimesh 5.x |
| S2 | 分片+类型 | 自研启发式（AABB/法向区域生长） | Open3D `segment_plane` |
| S3 | 拟合 | **NumPy/SciPy**（`least_squares`）+ 自研 RANSAC | **Eigen 3.4 + Ceres 2.0**（C++ 加速核） |
| S4 | 残差诊断 | SciPy `stats` + 鲁棒核（Huber/Tukey/95 分位） | — |
| S5 | MDL 停止/合并 | **自研**（`L_residual + L_model`） | — |
| S6 | 全局正则 | **SciPy `least_squares`** | **Ceres 2.0**（硬约束更强） |
| S7 | 拓扑构造 | **OCCT**（`BRepAlgoAPI` / `GeomAPI_IntSS` / `BRepBuilderAPI_*`） | OCP / pythonocc-core |
| S8 | 验证+导出 | **OCCT**（`BRepCheck` / `STEPControl_Writer` AP242）+ 欧拉校验 | — |
| — | 可视化 | **Open3D 交互窗口**（裸机原生 GL，v1 只能存 png） | PyVista、CloudCompare |
| — | 测试 | pytest | — |

> V1 与 v1 方案**技术栈不变**，仅可视化与 OCCT 获取方式因环境升级而放宽。

### 4.2 阶段二（V2）：深度学习栈 —— **训练本地化（5090 核心价值）**

| 算法阶段 | 首选 | 备选 | 训练显存 | 本地可行性 |
|---|---|---|---|---|
| P1 逐点类型/分割 | **PointTransformerV3（PTv3, Pointcept）** | PTv2 / PointNet++ / DGCNN | 数十 M 参数，批 8—16 约 **8—16 GB** | ✅ 5090 完全够 |
| P1 稀疏卷积对照 | **spconv 2.3（自编译 sm_120）** | MinkowskiEngine（**已淘汰，勿选**） | 同上 | ⚠️ 需编译，见 §5 |
| P2 类型判定头 | MLP/线性探针（借 latent） | 小分类器 | < 1 GB | ✅ |
| P3 边界性质头 | PTv3 共享骨干 + 多头 | 独立小 CNN | 同 P1 | ✅ |
| P5 latent 几何理解 | **ShapeVAE 冻结 encoder + 线性/MLP 探针** | 流匹配（Flow Matching） | 探针 CPU；VAE 训练 ~16 GB | ✅ |
| P5 生成式补全 | SDF/ShapeVAE decoder | 扩散模型 | 16—24 GB | ✅ |
| P5 LLM 提议 | **本地 7B—14B（量化）** | vLLM 本地/API | 7B-4bit ≈ 5 GB；14B-4bit ≈ 9 GB | ✅ 32 GB 富余 |

**训练框架**：PyTorch 2.7+（cu128）+ **Lightning**（结构化 trainer）或 **Accelerate**（轻量）+ **Hydra**（配置）+ **MLflow/W&B**（追踪）。5090 单卡足够训本框架 100 M 以内的模型。

**数据管线**（V1 计划已定「标签可程序化生成」）：
- 生成：CadQuery/OCP 程序化采样点云 + 类型/边界标签 → 存 **HDF5 / WebDataset / LMDB** 分片。
- 读取：`torch.utils.data` + `num_workers=8`（裸机可多进程）；5090 配高速 NVMe 避免 IO 饿死。
- 数据版本：**DVC**（大文件）或 `git-lfs`（小规模）。

### 4.3 阶段三（V3）：装配级 + 工程化平台

| 方向 | 首选 | 备选 |
|---|---|---|
| 部件关系推理 | **PyG（GNN）/ 小 Transformer** | 图神经 + 规则混合 |
| 装配约束求解 | **Ceres 2.0 / g2o / GTSAM**（因子图） | SciPy、CasADi |
| 大模型提议 | **本地 vLLM（7B—32B）** | OpenAI/Anthropic API |
| 程序化 CAD 验证 | **CadQuery**（把 LLM 输出当程序执行检验） | OpenSCAD |
| 主动消歧闭环 | **ROS 2 Humble**（机械臂/相机）+ 规划器 | 自研 |
| 流程编排 | **Snakemake / Prefect** | Airflow、自研 Makefile |
| 服务化 | **FastAPI + Docker** | gRPC |
| 可观测 | **MLflow + Prometheus/Grafana** | JSON provenance |
| 张量/几何交换 | **ONNX / glTF / STEP** | — |

> 有 32 GB 显存后，**V3 的本地 LLM 提议与 GNN 训练都可本地完成**，不再需要外部 API 或云算力。

---

## 五、Blackwell 兼容性深水区（本方案最大风险点，务必先打通）

### 5.1 spconv 与 `sm_120`（PTv3 的硬依赖）

`spconv` 的 PyPI 预编译 wheel 面向 cu118/cu120，**不含 Blackwell（`sm_120`）目标**；直接 `pip install spconv` 在 5090 上可能导入失败或运行时报 `no kernel image is available for execution on the device`。**解法：源码编译并显式指定架构。**

```bash
mamba activate wr
mamba install -y cuda-toolkit=12.8 -c nvidia      # 提供 nvcc
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0"                # ← Blackwell；Ada 用 8.9，Ampere 用 8.6
pip install ninja cmake
pip install spconv --no-binary spconv             # 触发源码构建（或 clone 后 pip install .）
# 或 CUDA 端底座
pip install cumm cumm-cu128                       # 需与 spconv 版本配套
python -c "import spconv;print(spconv.__version__)"
```

**若 spconv 编译受阻的降级路线**（按优先级）：
1. 用 **Pointcept 官方 Docker 镜像**（社区已适配较新 CUDA）作为训练容器；
2. 用**不依赖 spconv 的 attention 实现**（PTv3 的纯 attention 变体 / PTv2）做 V2 首版；
3. 暂以 **PointNet++/DGCNN（PyG，无稀疏卷积）** 起步，先跑通「DL 替换人工分片」的链路，再迭代骨干。

### 5.2 其他扩展的编译目标

| 库 | 风险 | 对策 |
|---|---|---|
| MinkowskiEngine | 对新 CUDA 支持差、基本停维 | **不选**，用 spconv |
| torch_scatter / torch_cluster（PyG） | 需匹配 CUDA 版本编译 | 用 PyG 官方 `pyg-lib` 预编译源（`-f https://data.pyg.org/whl/torch-2.7.0+cu128.html`） |
| flash-attn | 需编译、吃架构 | 装预编译 wheel 或用 PyTorch 内置 SDPA 兜底 |
| 混合精度 | Blackwell 支持 BF16/FP8 | `torch.autocast(dtype=torch.bfloat16)`；FP8 需 TransformerEngine |

### 5.3 验证「GPU 真的能用」的清单

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("cap", torch.cuda.get_device_capability(0), "name", torch.cuda.get_device_name(0))
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul ok", (a @ a).shape, "peakGB", torch.cuda.max_memory_allocated()/1e9)
PY
```

---

## 六、显存预算表（24 / 32 GB 能训什么）

| 模型 / 任务 | 参数量 | 批大小 | 精度 | 估算显存 | 24 GB | 32 GB |
|---|---|---|---|---|---|---|
| PTv3 分割（逐点，短序列） | ~30—50 M | 8 | BF16 | 8—14 GB | ✅ | ✅ |
| PTv3 分割（长序列/大场景） | 同上 | 16 | BF16 | 16—24 GB | ⚠️ 临界 | ✅ |
| ShapeVAE（latent） | 数十 M | 16 | FP32/BF16 | 12—20 GB | ✅ | ✅ |
| GNN 装配关系 | 小（<10 M） | 图批 | FP32 | 2—6 GB | ✅ | ✅ |
| SDF/扩散补全 | 数十—百 M | 8 | BF16 | 16—24 GB | ⚠️ | ✅ |
| 本地 LLM 推理 7B-4bit | 7 B | — | INT4 | ~5—6 GB | ✅ | ✅ |
| 本地 LLM 推理 14B-4bit | 14 B | — | INT4 | ~9—11 GB | ✅ | ✅ |
| 本地 LLM 推理 32B-4bit | 32 B | — | INT4 | ~18—20 GB | ⚠️ 临界 | ✅ |

> 本框架自估**可训练参数 < 100 M**（小于 ResNet-152，见框架 10.1），故 **5090 单卡对 V1—V3 全流程是充裕的**；显存不是瓶颈，**数据吞吐与 spconv 编译才是**。

---

## 七、新环境安装脚本（裸机 + 5090，可直接照做）

```bash
# 0) 系统与驱动（Ubuntu 22.04）
sudo apt update && sudo apt install -y build-essential git curl wget
sudo add-apt-repository -y ppa:graphics-drivers/ppa && sudo apt update
sudo apt install -y nvidia-driver-570           # Blackwell 最低门槛（以 NVIDIA 现行推荐为准）
sudo reboot
nvidia-smi                                      # 确认 5090 与驱动版本

# 1) CUDA + cuDNN（或用 mamba 装 cuda-toolkit，避免污染系统）
#    推荐：mamba 装 CUDA，系统只留驱动
./bin/micromamba install -n wr -y -c nvidia cuda-toolkit=12.8 cudnn=9

# 2) Python 环境
./bin/micromamba create -y -n wr python=3.12
./bin/micromamba activate wr

# 3) PyTorch（Blackwell 必须 cu128 及以上）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4) 几何 / 数值 / OCCT
pip install "numpy<3" scipy scikit-learn networkx open3d trimesh pyvista pytest
pip install cadquery                            # OCP/OCCT 绑定；或 mamba install pythonocc-core

# 5) C++ 性能核（S6 全局优化）
sudo apt install -y libeigen3-dev libceres-dev libcgal-dev

# 6) 训练与工程
pip install lightning hydra-core omegaconf mlflow dvc onnx onnxruntime
pip install torch-geometric
pip install -f https://data.pyg.org/whl/torch-2.7.0+cu128.html torch-scatter torch-cluster

# 7) spconv（按 §5.1 源码编译，指定 TORCH_CUDA_ARCH_LIST=12.0）

# 8) 验证
python -c "import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

---

## 八、与 v1 方案保持一致的部分（不因换机而改）

- **算法选型不变**：`docs/specs/algorithm-framework-and-selection.md` 的 P0—P8 与首选算法**全部保留**。
- **核心原则不变**：模型只输出**决策**（类型/分割/边界性质/候选），几何参数由**解析拟合**算出；传统几何是精度最终来源；拓扑先验强制执行，欧拉 `V−E+F−L=2(S−G)` 校验。
- **符号端不变**：OCCT（B-rep 构建 / STEP AP242）、Eigen/Ceres（拟合与全局正则）、MDL 与 GlobFit 自研。
- **目录结构不变**：`src/{io,segment,fit,diagnose,mdl,regopt,topology,verify,export,learn,assembly,eval}`。

---

## 九、一句话总结

> **换机后唯一实质变化是「训练从外置变本地」**：Blackwell（`sm_120`）要求 **驱动 ≥ 570 + CUDA 12.8+ + PyTorch 2.7/cu128**，并把 **spconv 源码编译（`TORCH_CUDA_ARCH_LIST=12.0`）** 作为 V2 的第一道工程关卡；显存 ≥24 GB（5090 实为 32 GB）对 <100 M 参数的本框架**绰绰有余**，瓶颈在数据吞吐与稀疏卷积编译，而非显存。传统几何栈（V1）与符号端（OCCT/Ceres）**完全不变**。
