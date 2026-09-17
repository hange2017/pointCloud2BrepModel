#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""min_feature_scale.py — 由「点云扫描精度」推算「最小可分辨特征尺度」。

为什么需要这个脚本
------------------
扫描精度（单点噪声 sigma，例如 0.5 mm）不是重建精度的上限：
  * 面的【位置】可借 N 个点平均提升到 sigma/sqrt(N)（微米级）；
  * 面的【法向】与【曲率】却受限于面片尺寸——小面片 / 窄特征上法向误差会急剧放大。
本脚本把「特征多小就测不准」量化出来，用于：
  1. 设定 V1 样本（S1/S2/S3）的特征尺寸下限；
  2. 标定各环节阈值 epsilon 的物理下限（避免把「传感器极限」误当「算法失败」）。

三个判据（分开列，不合并）
--------------------------
A. 法向估计（噪声主导）
   PCA 拟合半径 R 的圆盘、N 个点、法向噪声 sigma 时，法向角误差（解析近似）:
       theta_noise ≈ 2*sqrt(2) * sigma / (R * sqrt(N))     [rad]
   （推导：均匀圆盘 var(x)=R^2/4，两正交方向斜率误差合成；见 --verify 蒙特卡洛对照）
   反解给定误差预算 theta_budget 所需的最小面片半径：
       R_min = 2*sqrt(2) * sigma / (theta_budget * sqrt(N))

B. 采样带宽（Nyquist 式）
   任何特征沿表面的带宽 W 上至少要落 k 个采样点才能被辨识：
       W_min = k * spacing        （k 通常取 3~5）

C. 特征可分辨（信噪比 SNR，用于圆角 / 倒角判定）
   圆角半径 r 的 90° 弧：带宽 W = pi*r/2，带内点数 N_f = W/spacing，
   保守取等效圆盘半径 R_eff = W/2（特征窄，等效孔径小 → 法向更难估）：
       theta_noise(W) = 4*sqrt(2) * sigma * sqrt(spacing) / W**1.5     [rad]
       dtheta_feature = pi/2                                          [rad, 90° 弧]
       SNR = dtheta_feature / theta_noise
   要求 SNR >= snr_min（默认 5 视为可识别，10 视为稳健）：
       W_min_feature = ( snr_min * 8*sqrt(2) * sigma * sqrt(spacing) / pi ) ** (2/3)
       r_min = 2*W_min_feature/pi

用法
----
    python src/tools/min_feature_scale.py --sigma 0.5 --spacing 0.2
    python src/tools/min_feature_scale.py --sigma 0.5 --spacing 0.2 --verify
    python src/tools/min_feature_scale.py --sigma 0.1 --spacing 0.05 --snr-min 10
"""
from __future__ import annotations

import argparse
import math
import sys

# 让 Windows 控制台也能正确显示中文（cp936 默认会乱码）
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

SQRT2 = math.sqrt(2.0)
RAD2DEG = 180.0 / math.pi


# --------------------------------------------------------------------------
# 解析判据
# --------------------------------------------------------------------------
def theta_noise_rad(R: float, N: float, sigma: float) -> float:
    """判据 A：半径 R 圆盘上 N 点的 PCA 法向角误差（rad）。"""
    return 2.0 * SQRT2 * sigma / (R * math.sqrt(N))


def min_disk_radius(sigma: float, N: float, theta_budget_rad: float) -> float:
    """判据 A 反解：给定法向误差预算，所需最小面片半径。"""
    return 2.0 * SQRT2 * sigma / (theta_budget_rad * math.sqrt(N))


def min_feature_width(sigma: float, spacing: float, snr_min: float) -> float:
    """判据 C 反解：给定 SNR 门槛，可分辨的最小特征带宽（mm）。"""
    return (snr_min * 8.0 * SQRT2 * sigma * math.sqrt(spacing) / math.pi) ** (2.0 / 3.0)


# --------------------------------------------------------------------------
# 蒙特卡洛验证（--verify），仅用于核对判据 A 的解析近似
# --------------------------------------------------------------------------
def verify_theta_noise(sigma: float, trials: int = 300, seed: int = 42) -> None:
    import numpy as np

    rng = np.random.default_rng(seed)
    print("\n[verify] 蒙特卡洛 vs 解析式 —— 判据 A 法向角误差 (deg), 中位数")
    hdr = "R\\N"
    print(f"{hdr:>8}" + "".join(f"{n:>12}" for n in (500, 2000, 10000)))
    for R in (3.0, 5.0, 10.0, 20.0):
        cells = []
        for N in (500, 2000, 10000):
            errs = []
            for _ in range(trials):
                rr = R * np.sqrt(rng.random(N))
                tt = rng.random(N) * 2 * np.pi
                pts = np.stack([rr * np.cos(tt), rr * np.sin(tt), rng.normal(0, sigma, N)], 1)
                pts -= pts.mean(0)
                _, _, vt = np.linalg.svd(pts, full_matrices=False)
                ang = math.degrees(math.acos(min(1.0, abs(float(vt[2][2])))))
                errs.append(ang)
            mc = float(np.median(errs))
            an = theta_noise_rad(R, N, sigma) * RAD2DEG
            cells.append(f"{mc:5.3f}/{an:5.3f}")
        print(f"{R:>6.0f}mm" + "".join(f"{c:>12}" for c in cells))
    print("（每格 = 蒙特卡洛中位数 / 解析式估计；两者同量级即认为近似可用）")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="由扫描精度推算最小可分辨特征尺度",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--sigma", type=float, default=0.5, help="单点扫描噪声 sigma（mm）")
    ap.add_argument("--spacing", type=float, default=0.2, help="相邻采样点平均间距（mm）")
    ap.add_argument("--k", type=int, default=5, help="判据 B：特征带宽上最少采样点数")
    ap.add_argument("--theta-budget", type=float, default=1.0, help="判据 A：法向误差预算（deg）")
    ap.add_argument("--snr-min", type=float, default=5.0, help="判据 C：识别 SNR 门槛（>=5 可识别，>=10 稳健）")
    ap.add_argument("--verify", action="store_true", help="蒙特卡洛验证判据 A 的解析近似")
    args = ap.parse_args()

    sigma, s = args.sigma, args.spacing
    budget = args.theta_budget / RAD2DEG

    print("=" * 74)
    print("最小可分辨特征尺度推算")
    print("=" * 74)
    print(f"输入：扫描噪声 sigma = {sigma} mm，采样间距 spacing = {s} mm")
    print(f"      k = {args.k} 点，法向误差预算 = {args.theta_budget}°，SNR 门槛 = {args.snr_min}")

    # ---- 判据 A：法向误差表 ----
    print("\n[判据 A] 法向角误差中位数（deg）：theta ≈ 2√2·sigma/(R√N)")
    Ns = (500, 2000, 10000, 50000)
    hdr = "R\\N"
    print(f"{hdr:>8}" + "".join(f"{n:>12}" for n in Ns))
    for R in (3, 5, 10, 20, 50):
        row = "".join(f"{theta_noise_rad(R, n, sigma) * RAD2DEG:12.3f}" for n in Ns)
        print(f"{R:>6}mm{row}")

    print(f"\n[判据 A] 达到 {args.theta_budget}° 预算所需的最小面片半径 R_min（mm）：")
    for n in Ns:
        print(f"    N = {n:>6d} 点  ->  R_min = {min_disk_radius(sigma, n, budget):6.2f} mm")

    # ---- 判据 B：采样带宽 ----
    w_b = args.k * s
    print(f"\n[判据 B] 采样带宽：W_min = k·spacing = {args.k} × {s} = {w_b:.2f} mm")

    # ---- 判据 C：圆角可分辨 ----
    w_c = min_feature_width(sigma, s, args.snr_min)
    r_c = 2.0 * w_c / math.pi
    w_c10 = min_feature_width(sigma, s, 10.0)
    r_c10 = 2.0 * w_c10 / math.pi
    print(f"\n[判据 C] 圆角（90°弧）可分辨下限：")
    print(f"    SNR >= {args.snr_min:<4} ->  最小带宽 {w_c:5.2f} mm  =>  最小圆角半径 r_min ≈ {r_c:.2f} mm")
    print(f"    SNR >= 10  （稳健） ->  最小带宽 {w_c10:5.2f} mm  =>  最小圆角半径 r_min ≈ {r_c10:.2f} mm")

    # ---- 汇总 ----
    r_a = min_disk_radius(sigma, 2000, budget)  # 以典型 2000 点面片为代表
    print("\n" + "-" * 74)
    print("结论（供 V1 阈值标定与样本尺寸设定）")
    print("-" * 74)
    print(f"  1) 面【位置】精度无关特征尺度：N 点平均后 ≈ sigma/√N，"
          f"如 N=2000 -> {sigma / math.sqrt(2000) * 1000:.1f} µm")
    print(f"  2) 面【法向】需足够大的面片：2000 点下要达 {args.theta_budget}°，面片半径 R >= {r_a:.2f} mm")
    print(f"  3) 特征【带宽】下限（采样）：W >= {w_b:.2f} mm")
    print(f"  4) 圆角/相切【可分辨半径】：r >= {r_c:.2f} mm（SNR≥{args.snr_min:g}），"
          f"r >= {r_c10:.2f} mm（SNR≥10 稳健）")
    print(f"  5) 综合建议：V1 样本的特征尺度（圆角半径 / 倒角宽度 / 孔径）"
          f"建议 >= {max(r_c10, w_b):.1f} mm；低于 {r_c:.1f} mm 的特征应预期失败，"
          f"且属【传感器极限】而非算法缺陷")
    print("-" * 74)

    if args.verify:
        verify_theta_noise(sigma)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
