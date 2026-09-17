# pointCloud2BrepModel

多视角点云 → **装配级 B-rep（As-Built 逆向 CAD）** 重建的调研结论与工程落地方案。

## 项目定位

输入：位姿已知的多视角稠密点云（相机可挂机械臂做主动补拍）。
输出：装配级、带不确定性标注的 As-Built B-rep / STEP AP242 模型。

核心范式是 **「DL 感知 + 传统几何求解」的混合（Hybrid）**，三条硬约束：

1. 网络只输出**决策**（面类型 / 分割 / 边界性质 / 候选），**几何参数一律由解析拟合算出**（禁止直接回归几何坐标）；
2. 精度最终落在符号端——SSI 求交、裁剪、约束优化、欧拉校验、STEP 导出；
3. 拓扑先验（B-rep 合法性与欧拉公式约束）在构造阶段强制执行，欠定处交给人或选择机制兜底。

## 目录结构

| 路径 | 内容 |
|---|---|
| `docs/specs/algorithm-framework-and-selection.md` | 工程实现方案：P0—P8 主流程 + 每阶段 3—5 个算法选型（按本场景预期效果排序） |
| `docs/specs/platform-and-toolchain.md` | 平台与开发工具链方案（三环境）：本机开发（Windows + GTX 1080）/ 训练（A6000）/ 文档，含已实测通过的环境基线 |
| `docs/notes/index.md` | 跨文档综合索引：总体框架 × 拓扑同胚 的章节交叉对照 |
| `docs/notes/framework-*.md` | 总体框架（第 0—11 章）结构化笔记 |
| `docs/notes/topology-notes.md` | 拓扑同胚与 latent 可行性论证的结构化笔记 |
| `docs/*.docx` | 原始调研文档（总体框架、拓扑同胚 / latent） |
| `.dsb/` | 项目约定与 agent 工作区（指令、计划、设计说明）；`skills/` 为工具自带技能包，已在 `.gitignore` 中排除 |

约定：实现计划写到 `.dsb/plans/`，设计说明写到 `.dsb/specs/`，其它文档写到 `.dsb/docs/`。

## 环境基线（摘要）

项目分**三个环境**（详见 `docs/specs/platform-and-toolchain.md`）：

| 环境 | 硬件 | 用途 | 状态 |
|---|---|---|---|
| **本机开发** | Windows 10 · GTX 1080（`sm_61`, 8 GB） | 写代码、V1 纯几何链路、测试 | ✅ 已搭好并实测通过 |
| **训练** | 公司 A6000（`sm_86`, 48 GB）· Linux | V2/V3 的 DL 训练 | ⏳ 待机器就绪 |
| **文档** | 任意 | 本仓库文档 | — |

本机环境 `p2b`（Python 3.11.16，装于 `E:` 盘）已锁定为：`open3d 0.20.0` + `cadquery 2.8.0`（OCCT 7.9.3.1.1）+ `torch 2.5.1+cu121`（CUDA build 12.1）+ 数值栈。复现与验证：

```bat
powershell -ExecutionPolicy Bypass -File envs\setup-p2b.ps1
conda run --no-capture-output -n p2b python envs\verify_env.py
```

> **口径说明**：本机为 Pascal 架构，PyTorch 自 2.8 起移除 Pascal 支持，故本机 torch **锁死在 `2.5.1+cu121`**（物理上限，非保守选择）；训练放到 A6000。下载统一走国内镜像（清华 PyPI + SJTU PyTorch，实测 38—70 MB/s），不依赖 VPN。

## V1 样本与精度约束（摘要）

按 `.dsb/plans/v1-minimal-loop.md`：V1 用**三级递进样本 S1/S2/S3**（S1 长方体顶点零观测 → **S2 带孔板+螺钉（核心）** → S3 倒角/圆角/相切）。扫描精度 0.5 mm 下，特征尺度下限由 `src/tools/min_feature_scale.py` 计算：**V1 样本特征建议 ≥ 2.6 mm，低于 1.6 mm 预期失败**（传感器物理极限，非算法缺陷）。

## 使用

```bash
git clone <repo-url>
cd pointCloud2BrepModel
```

首次提交前请设置提交身份（仓库级即可）：

```bash
git config user.name "你的名字"
git config user.email "你的邮箱"
```

> 说明：`.gitignore` 默认排除点云 / CAD 模型等大文件（`*.ply`、`*.step`、`*.pt` 等）。若需把小型样例数据入库，删除对应行或追加 `!样例文件` 白名单即可。
