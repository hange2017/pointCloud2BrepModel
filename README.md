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
| `docs/specs/platform-and-toolchain.md` | 平台与开发工具链方案：裸机 + RTX 5090 的驱动 / CUDA / PyTorch / spconv 版本矩阵 |
| `docs/notes/index.md` | 跨文档综合索引：总体框架 × 拓扑同胚 的章节交叉对照 |
| `docs/notes/framework-*.md` | 总体框架（第 0—11 章）结构化笔记 |
| `docs/notes/topology-notes.md` | 拓扑同胚与 latent 可行性论证的结构化笔记 |
| `docs/*.docx` | 原始调研文档（总体框架、拓扑同胚 / latent） |
| `.dsb/` | 项目约定与 agent 工作区（指令、计划、设计说明）；`skills/` 为工具自带技能包，已在 `.gitignore` 中排除 |

约定：实现计划写到 `.dsb/plans/`，设计说明写到 `.dsb/specs/`，其它文档写到 `.dsb/docs/`。

## 环境基线（摘要）

按 `docs/specs/platform-and-toolchain.md`：Ubuntu 22.04 + NVIDIA 驱动 ≥ 570 + CUDA Toolkit 12.8+（`sm_120`）+ PyTorch 2.7+（`cu128` 轮子）+ mamba 管理 Python 3.11/3.12；`pythonocc-core` 走 conda 安装，`spconv` 需针对 Blackwell 自编译。

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
