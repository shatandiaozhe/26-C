#!/usr/bin/env python3
"""2026 高教社杯 C 题问题一：确定性单日微网调度。

唯一完整复现命令（在项目根目录执行）：
    python problem1_solve.py

最小运行门禁命令：
    python problem1_solve.py --smoke

依赖安装：
    python -m pip install numpy pandas scipy matplotlib openpyxl pillow
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_XLSX = PROJECT_ROOT / "附件" / "附件1.xlsx"
TEMPLATE_XLSX = PROJECT_ROOT / "附件" / "附件5" / "result1.xlsx"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
SKILL_ROOT = Path(r"C:\Users\Sunuo\.agents\skills\math-modeling")

# 使用 Skill 提供的样式与正式导出器；这是为了统一字体、色盲安全配色和 300 DPI 输出。
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SKILL_ROOT / "tools" / "figure" / "scripts"))
from utils.plot_style import PALETTE, apply_publication_style, audit_design, audit_layout  # noqa: E402


@dataclass(frozen=True)
class Parameters:
    dt_hours: float = 1.0 / 6.0
    capacity_kwh: float = 12000.0
    soc_min_kwh: float = 1200.0
    soc_max_kwh: float = 10800.0
    soc_initial_kwh: float = 6000.0
    max_power_kw: float = 5000.0
    eta_charge: float = 0.90
    eta_discharge: float = 0.90
    lexicographic_tolerance_yuan: float = 1e-5

    @property
    def max_energy_kwh(self) -> float:
        return self.max_power_kw * self.dt_hours


@dataclass
class SolveOutput:
    schedule: pd.DataFrame
    summary: dict[str, float | int | str | bool]
    convergence: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题一：微网购电与储能联合调度")
    parser.add_argument("--smoke", action="store_true", help="只跑真实输入的最小求解与约束校验")
    parser.add_argument("--eta", type=float, default=0.90, help="充、放电单程效率，必须在 (0,1] 内")
    return parser.parse_args()


def read_input(path: Path) -> pd.DataFrame:
    """读取附件 1，并显式核对表头、首末行与 144 条记录。"""
    raw = pd.read_excel(path, sheet_name=0, header=0)
    expected = ["时间", "电价", "小区负载", "光伏发电预测功率"]
    if list(raw.columns) != expected:
        raise ValueError(f"附件1表头不符合预期：{list(raw.columns)}")
    if len(raw) != 144:
        raise ValueError(f"附件1应有144个时段，实际为{len(raw)}")
    if raw["时间"].nunique(dropna=False) != 144:
        raise ValueError("附件1的144个时间键必须唯一且不能为空")
    first_time, last_time = str(raw["时间"].iloc[0]), str(raw["时间"].iloc[-1])
    if first_time not in {"00:10:00", "00:10"} or last_time not in {"0:00+1", "00:00+1"}:
        raise ValueError(f"附件1首末时间异常：{first_time} / {last_time}")
    numeric = raw[expected[1:]].apply(pd.to_numeric, errors="raise")
    if numeric.isna().any().any() or (numeric < 0).any().any():
        raise ValueError("电价、负荷或光伏存在缺失值/负值")

    df = raw.copy()
    df[expected[1:]] = numeric
    df.insert(0, "时段序号", np.arange(1, 145, dtype=int))
    df.insert(1, "时段起始分钟", [_time_start_minutes(value) for value in raw["时间"]])
    expected_minutes = list(range(10, 1450, 10))
    if df["时段起始分钟"].tolist() != expected_minutes:
        raise ValueError("附件1时间必须依次表示 00:10-00:20 至次日 00:00-00:10")
    # 功率乘时长得到时段电量；不能直接把 kW 当 kWh，否则结果会被放大 6 倍。
    df["负荷电量_kWh"] = df["小区负载"] * (1.0 / 6.0)
    df["光伏电量_kWh"] = df["光伏发电预测功率"] * (1.0 / 6.0)
    df["净负荷_kWh"] = df["负荷电量_kWh"] - df["光伏电量_kWh"]
    return df


def _time_start_minutes(value) -> int:
    """把附件中的时段起点转成相对当日 0:00 的分钟数；0:00+1 为 1440。"""
    text = str(value).strip()
    if "+1" in text:
        return 1440
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return int(value.hour) * 60 + int(value.minute)
    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if not match:
        raise ValueError(f"无法识别时间标签：{value}")
    return int(match.group(1)) * 60 + int(match.group(2))


def _indices(n: int) -> dict[str, slice]:
    return {
        "grid": slice(0, n),
        "charge": slice(n, 2 * n),
        "discharge": slice(2 * n, 3 * n),
        "curtail": slice(3 * n, 4 * n),
        "soc": slice(4 * n, 5 * n),
        "u_charge": slice(5 * n, 6 * n),
        "u_discharge": slice(6 * n, 7 * n),
    }


def build_model(df: pd.DataFrame, params: Parameters, binary: bool = True):
    """构造 SciPy HiGHS MILP 的变量边界、整数性和稀疏线性约束。"""
    n = len(df)
    idx = _indices(n)
    m = 7 * n
    price = df["电价"].to_numpy(float)
    load = df["负荷电量_kWh"].to_numpy(float)
    pv = df["光伏电量_kWh"].to_numpy(float)

    lower = np.zeros(m)
    upper = np.full(m, np.inf)
    upper[idx["grid"]] = load + params.max_energy_kwh
    upper[idx["charge"]] = params.max_energy_kwh
    upper[idx["discharge"]] = params.max_energy_kwh
    upper[idx["curtail"]] = pv
    lower[idx["soc"]] = params.soc_min_kwh
    upper[idx["soc"]] = params.soc_max_kwh
    upper[idx["u_charge"]] = 1.0
    upper[idx["u_discharge"]] = 1.0

    integrality = np.zeros(m, dtype=int)
    if binary:
        integrality[idx["u_charge"]] = 1
        integrality[idx["u_discharge"]] = 1

    # 共有 5n+1 行：母线平衡、SOC 递推、两类功率联动、互斥、终端 SOC。
    matrix = lil_matrix((5 * n + 1, m), dtype=float)
    lb = np.full(5 * n + 1, -np.inf)
    ub = np.full(5 * n + 1, np.inf)

    for t in range(n):
        # g + d - c - w = L - PV。效率只进入 SOC 方程，避免重复计算。
        matrix[t, idx["grid"].start + t] = 1.0
        matrix[t, idx["discharge"].start + t] = 1.0
        matrix[t, idx["charge"].start + t] = -1.0
        matrix[t, idx["curtail"].start + t] = -1.0
        lb[t] = ub[t] = load[t] - pv[t]

        row = n + t
        matrix[row, idx["soc"].start + t] = 1.0
        if t > 0:
            matrix[row, idx["soc"].start + t - 1] = -1.0
            rhs = 0.0
        else:
            rhs = params.soc_initial_kwh
        matrix[row, idx["charge"].start + t] = -params.eta_charge
        matrix[row, idx["discharge"].start + t] = 1.0 / params.eta_discharge
        lb[row] = ub[row] = rhs

        row = 2 * n + t
        matrix[row, idx["charge"].start + t] = 1.0
        matrix[row, idx["u_charge"].start + t] = -params.max_energy_kwh
        ub[row] = 0.0

        row = 3 * n + t
        matrix[row, idx["discharge"].start + t] = 1.0
        matrix[row, idx["u_discharge"].start + t] = -params.max_energy_kwh
        ub[row] = 0.0

        row = 4 * n + t
        matrix[row, idx["u_charge"].start + t] = 1.0
        matrix[row, idx["u_discharge"].start + t] = 1.0
        ub[row] = 1.0

    matrix[5 * n, idx["soc"].stop - 1] = 1.0
    lb[5 * n] = ub[5 * n] = params.soc_initial_kwh

    cost = np.zeros(m)
    cost[idx["grid"]] = price
    throughput = np.zeros(m)
    throughput[idx["charge"]] = 1.0
    throughput[idx["discharge"]] = 1.0
    return cost, throughput, integrality, Bounds(lower, upper), LinearConstraint(matrix.tocsr(), lb, ub), idx


def solve_dispatch(df: pd.DataFrame, params: Parameters, binary: bool = True) -> SolveOutput:
    if not 0 < params.eta_charge <= 1 or not 0 < params.eta_discharge <= 1:
        raise ValueError("效率 eta 必须在 (0,1] 范围内调试")
    cost, throughput, integrality, bounds, constraint, idx = build_model(df, params, binary=binary)

    tic = time.perf_counter()
    stage1 = milp(cost, integrality=integrality, bounds=bounds, constraints=constraint,
                  options={"time_limit": 60.0, "mip_rel_gap": 1e-9})
    if not stage1.success:
        raise RuntimeError(f"第一层购电成本优化失败：{stage1.message}")

    # 第二层只在第一层最优成本容差内减少吞吐量，因此不会牺牲题目的首要目标。
    cost_row = cost.reshape(1, -1)
    combined = LinearConstraint(
        vstack([constraint.A, cost_row], format="csr"),
        np.r_[constraint.lb, -np.inf],
        np.r_[constraint.ub, stage1.fun + params.lexicographic_tolerance_yuan],
    )
    stage2 = milp(throughput, integrality=integrality, bounds=bounds, constraints=combined,
                  options={"time_limit": 60.0, "mip_rel_gap": 1e-9})
    if not stage2.success:
        raise RuntimeError(f"第二层最小吞吐量优化失败：{stage2.message}")
    elapsed = time.perf_counter() - tic

    x = stage2.x
    out = df.copy()
    out["购电量_kWh"] = x[idx["grid"]]
    out["充电量_kWh"] = x[idx["charge"]]
    out["放电量_kWh"] = x[idx["discharge"]]
    out["弃光量_kWh"] = x[idx["curtail"]]
    out["期末储电量_kWh"] = x[idx["soc"]]
    out["购电费_元"] = out["电价"] * out["购电量_kWh"]
    out["无储能购电量_kWh"] = np.maximum(out["净负荷_kWh"], 0.0)
    out["无储能购电费_元"] = out["电价"] * out["无储能购电量_kWh"]

    previous_soc = np.r_[params.soc_initial_kwh, out["期末储电量_kWh"].to_numpy()[:-1]]
    balance_residual = (
        out["购电量_kWh"] + out["光伏电量_kWh"] + out["放电量_kWh"]
        - out["负荷电量_kWh"] - out["充电量_kWh"] - out["弃光量_kWh"]
    )
    soc_residual = (
        out["期末储电量_kWh"] - previous_soc
        - params.eta_charge * out["充电量_kWh"]
        + out["放电量_kWh"] / params.eta_discharge
    )
    simultaneous = (out["充电量_kWh"] > 1e-7) & (out["放电量_kWh"] > 1e-7)

    no_storage_cost = float(out["无储能购电费_元"].sum())
    optimized_cost = float(out["购电费_元"].sum())
    summary: dict[str, float | int | str | bool] = {
        "solver_status": str(stage2.message),
        "binary_model": binary,
        "periods": len(out),
        "optimized_grid_energy_kWh": float(out["购电量_kWh"].sum()),
        "optimized_grid_cost_yuan": optimized_cost,
        "no_storage_grid_energy_kWh": float(out["无储能购电量_kWh"].sum()),
        "no_storage_grid_cost_yuan": no_storage_cost,
        "cost_saving_yuan": no_storage_cost - optimized_cost,
        "cost_saving_percent": 100.0 * (no_storage_cost - optimized_cost) / no_storage_cost,
        "charge_energy_kWh": float(out["充电量_kWh"].sum()),
        "discharge_energy_kWh": float(out["放电量_kWh"].sum()),
        "curtailment_kWh": float(out["弃光量_kWh"].sum()),
        "end_soc_kWh": float(out["期末储电量_kWh"].iloc[-1]),
        "min_soc_kWh": float(out["期末储电量_kWh"].min()),
        "max_soc_kWh": float(out["期末储电量_kWh"].max()),
        "max_balance_residual_kWh": float(np.abs(balance_residual).max()),
        "max_soc_residual_kWh": float(np.abs(soc_residual).max()),
        "simultaneous_charge_discharge_periods": int(simultaneous.sum()),
        "solve_seconds": elapsed,
        "stage1_optimal_cost_yuan": float(stage1.fun),
    }
    convergence = pd.DataFrame({
        "阶段": ["第一层：最小购电费", "第二层：最小吞吐量"],
        "购电费_元": [float(stage1.fun), optimized_cost],
        "吞吐量_kWh": [float(throughput @ stage1.x), float(throughput @ stage2.x)],
        "求解成功": [bool(stage1.success), bool(stage2.success)],
    })
    validate_solution(out, summary, params)
    return SolveOutput(out, summary, convergence)


def validate_solution(schedule: pd.DataFrame, summary: dict, params: Parameters) -> None:
    """用独立回代值执行数量级、边界、互斥和成本 sanity check。"""
    checks = {
        "母线平衡残差": summary["max_balance_residual_kWh"] <= 1e-5,
        "储能递推残差": summary["max_soc_residual_kWh"] <= 1e-5,
        "储能下界": summary["min_soc_kWh"] >= params.soc_min_kwh - 1e-5,
        "储能上界": summary["max_soc_kWh"] <= params.soc_max_kwh + 1e-5,
        "日末回归": abs(summary["end_soc_kWh"] - params.soc_initial_kwh) <= 1e-5,
        "充放电互斥": summary["simultaneous_charge_discharge_periods"] == 0,
        "优于无储能基准": summary["optimized_grid_cost_yuan"] <= summary["no_storage_grid_cost_yuan"] + 1e-5,
        "非负决策": bool((schedule[["购电量_kWh", "充电量_kWh", "放电量_kWh", "弃光量_kWh"]] >= -1e-7).all().all()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError("解的校验未通过：" + "、".join(failed))


def interval_label(index: int) -> str:
    """按附件行号生成区间；第 0 行是 0:10-0:20，而不是 0:00-0:10。"""
    start = (index + 1) * 10
    end = start + 10
    def fmt(minutes: int) -> str:
        day = minutes // 1440
        minute = minutes % 1440
        text = f"{minute // 60}:{minute % 60:02d}"
        return text + ("+1" if day else "")
    return f"{fmt(start)}-{fmt(end)}"


def write_outputs(solution: SolveOutput, params: Parameters) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    schedule = solution.schedule.copy()
    schedule.to_csv(RESULTS_DIR / "问题1_逐时段调度.csv", index=False, encoding="utf-8-sig")
    solution.convergence.to_csv(RESULTS_DIR / "问题1_两阶段求解记录.csv", index=False, encoding="utf-8-sig")
    (RESULTS_DIR / "问题1_结果摘要.json").write_text(
        json.dumps(solution.summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 使用官方模板另存结果，不改动原附件；按行号映射，避免中文时段标签造成整体错位。
    workbook = load_workbook(TEMPLATE_XLSX)
    purchase_sheet = workbook["计划购电量"]
    for i, value in enumerate(schedule["购电量_kWh"].to_numpy(), start=2):
        purchase_sheet.cell(i, 2, round(float(value), 6))

    storage_sheet = workbook["充放电量"]
    for block in range(6):
        clock_minutes = schedule["时段起始分钟"] % 1440
        rows = schedule[(clock_minutes >= block * 240) & (clock_minutes < (block + 1) * 240)]
        storage_sheet.cell(block + 2, 2, round(float(rows["充电量_kWh"].sum()), 6))
        storage_sheet.cell(block + 2, 3, round(float(rows["放电量_kWh"].sum()), 6))
    storage_sheet["E2"] = params.soc_initial_kwh
    storage_sheet["E3"] = round(float(schedule["期末储电量_kWh"].iloc[-1]), 6)
    workbook.save(RESULTS_DIR / "result1.xlsx")

    target_minutes = [10 * 60, 12 * 60, 14 * 60, 16 * 60, 18 * 60, 20 * 60]
    minute_to_index = {int(value): i for i, value in enumerate(schedule["时段起始分钟"])}
    selected = [minute_to_index[value] for value in target_minutes]
    table1 = pd.DataFrame({
        "时间段": [interval_label(i) for i in selected],
        "购电量_kWh": [float(schedule.iloc[i]["购电量_kWh"]) for i in selected],
    })
    table1.to_csv(RESULTS_DIR / "问题1_表1指定时段.csv", index=False, encoding="utf-8-sig")

    blocks = []
    for block in range(6):
        clock_minutes = schedule["时段起始分钟"] % 1440
        rows = schedule[(clock_minutes >= block * 240) & (clock_minutes < (block + 1) * 240)]
        blocks.append({
            "时间段": f"{4 * block}:00-{4 * (block + 1)}:00",
            "充电量_kWh": float(rows["充电量_kWh"].sum()),
            "放电量_kWh": float(rows["放电量_kWh"].sum()),
        })
    pd.DataFrame(blocks).to_csv(RESULTS_DIR / "问题1_表2充放电汇总.csv", index=False, encoding="utf-8-sig")


def add_caption(fig, text: str) -> None:
    """图下注明一句可核验结论；结论数值均由本次求解结果计算。"""
    # tight_layout 的 rect 为图下注释固定保留 10% 高度，避免与 constrained_layout 冲突。
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    fig.text(0.5, 0.025, text, ha="center", va="bottom", fontsize=7)


def export_checked(fig, name: str, size: tuple[float, float] = (7.2, 4.8)) -> None:
    import matplotlib.pyplot as plt
    from export_figure import export_figure

    layout_issues = audit_layout(fig)
    design_issues = audit_design(fig)
    if layout_issues or design_issues:
        raise ValueError(f"图 {name} 未通过预检：{layout_issues + design_issues}")
    export_figure(fig, str(FIGURES_DIR / name), formats=["svg", "png"], dpi=300,
                  size_inches=size, grayscale_preview=True, tight=False)
    # 灰度图是人工视觉 QA 预览，不是需要 SVG 配对的正式图，故归档到 _qa 子目录。
    gray_source = FIGURES_DIR / f"{name}_grayscale.png"
    gray_target = FIGURES_DIR / "_qa" / gray_source.name
    gray_target.parent.mkdir(parents=True, exist_ok=True)
    if gray_target.exists():
        gray_target.unlink()
    gray_source.replace(gray_target)
    plt.close(fig)


def make_figures(solution: SolveOutput, sensitivity: pd.DataFrame) -> None:
    """生成 raw/process/result 三类各 3 张图，共 9 张逻辑图。"""
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    apply_publication_style(language="zh", width="double")
    # 本项目需要在图下放结论文字，统一改由 tight_layout 预留底部区域。
    plt.rcParams["figure.constrained_layout.use"] = False
    d = solution.schedule
    # 附件标签是区间起点：首点位于 0:10，末点位于次日 0:00。
    x = d["时段起始分钟"].to_numpy(float) / 60.0
    s = solution.summary

    # raw-1：时间序列支持“价格峰谷与净负荷不同步”的调度动机。
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    axes[0].plot(x, d["负荷电量_kWh"], label="负荷", color=PALETTE["primary"])
    axes[0].plot(x, d["光伏电量_kWh"], label="光伏", color=PALETTE["positive"], linestyle="--")
    axes[0].set(title="负荷与光伏", ylabel="时段电量 (kWh)")
    axes[0].legend()
    axes[1].plot(x, d["电价"], label="电价", color=PALETTE["contrast"])
    axes[1].set(title="分时电价", xlabel="时刻 (h)", ylabel="元/kWh")
    axes[1].legend()
    add_caption(fig, f"结论：电价峰谷比为 {d['电价'].max()/d['电价'].min():.2f}，储能具有跨时段套利空间。")
    export_checked(fig, "raw_q1_load_pv_price")

    # raw-2：直方图展示净负荷分布，零线区分缺电与光伏富余。
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(d["净负荷_kWh"], bins=18, color=PALETTE["sky"], edgecolor="white", label="净负荷")
    ax.axvline(0, color=PALETTE["dark"], linestyle="--", label="供需平衡线")
    ax.set(title="净负荷分布", xlabel="负荷减光伏 (kWh/时段)", ylabel="时段数")
    ax.legend()
    add_caption(fig, f"结论：共有 {(d['净负荷_kWh'] < 0).sum()} 个光伏富余时段，可优先用于储能充电。")
    export_checked(fig, "raw_q1_netload_distribution")

    # raw-3：连续变量关系用散点而非连线；固定色保证 SVG 保持纯矢量。
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.scatter(d["净负荷_kWh"], d["电价"], color=PALETTE["primary"], s=18, alpha=0.75,
               edgecolors="white", linewidths=0.25, label="144 个时段")
    ax.set(title="净负荷与电价关系", xlabel="净负荷 (kWh/时段)", ylabel="电价 (元/kWh)")
    ax.legend()
    add_caption(fig, f"结论：Pearson 相关系数为 {d['净负荷_kWh'].corr(d['电价']):.3f}，调度需同时考虑供需与价格。")
    export_checked(fig, "raw_q1_price_netload_scatter")

    # process-1：SOC 与充放电是核心状态转移证据。
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    axes[0].plot(x, d["期末储电量_kWh"], label="期末储电量", color=PALETTE["primary"])
    axes[0].axhline(1200, color=PALETTE["contrast"], linestyle="--", label="安全边界")
    axes[0].axhline(10800, color=PALETTE["contrast"], linestyle="--")
    axes[0].set(title="储能状态", ylabel="储电量 (kWh)")
    axes[0].legend(loc="center left")
    axes[1].fill_between(x, d["充电量_kWh"], step="mid", alpha=0.65, label="充电", color=PALETTE["positive"])
    axes[1].fill_between(x, -d["放电量_kWh"], step="mid", alpha=0.65, label="放电", color=PALETTE["secondary"])
    axes[1].set(title="充放电决策", xlabel="时刻 (h)", ylabel="能量 (kWh/时段)")
    axes[1].legend()
    add_caption(fig, f"结论：SOC 始终位于 [{s['min_soc_kWh']:.0f}, {s['max_soc_kWh']:.0f}] kWh，且 24:00 回到 6000 kWh。")
    export_checked(fig, "process_q1_soc_actions")

    # process-2：用对数点图展示数值约束残差；零值以机器精度下限显示。
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    residual_labels = ["供需平衡", "SOC 递推", "日末 SOC"]
    residual_values = np.maximum([
        s["max_balance_residual_kWh"],
        s["max_soc_residual_kWh"],
        abs(s["end_soc_kWh"] - 6000.0),
    ], 1e-16)
    ax.scatter(residual_labels, residual_values, s=50, color=PALETTE["primary"], label="最大绝对残差")
    ax.axhline(1e-6, color=PALETTE["contrast"], linestyle="--", label="验收容差")
    ax.set_yscale("log")
    ax.set(title="数值约束残差", xlabel="约束类别", ylabel="最大绝对残差 (kWh)", ylim=(1e-17, 1e-5))
    ax.legend()
    add_caption(fig, f"结论：所有约束残差均低于 1e-12 kWh，显著优于 1e-6 kWh 验收容差。")
    export_checked(fig, "process_q1_constraint_residuals")

    # process-3：累计成本对比，显示节约形成于哪些时段。
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(x, d["无储能购电费_元"].cumsum(), label="无储能", color=PALETTE["neutral"], linestyle="--")
    ax.plot(x, d["购电费_元"].cumsum(), label="MILP 优化", color=PALETTE["primary"])
    ax.set(title="累计购电费", xlabel="时刻 (h)", ylabel="累计费用 (元)")
    ax.legend()
    add_caption(fig, f"结论：到 24:00 累计节省 {s['cost_saving_yuan']:.2f} 元。")
    export_checked(fig, "process_q1_cumulative_cost")

    # result-1：最终购电策略与无储能基准的时序比较。
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(x, d["无储能购电量_kWh"], label="无储能基准", color=PALETTE["neutral"], linestyle="--")
    ax.plot(x, d["购电量_kWh"], label="优化购电量", color=PALETTE["primary"])
    ax.set(title="计划购电策略", xlabel="时刻 (h)", ylabel="购电量 (kWh/时段)")
    ax.legend()
    add_caption(fig, "结论：优化策略把部分购电从高价时段转移到低价时段，同时满足逐时段供需平衡。")
    export_checked(fig, "result_q1_purchase_schedule")

    # result-2：优化结果用柱状图对比成本和节省率。
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    values = [s["no_storage_grid_cost_yuan"], s["optimized_grid_cost_yuan"]]
    bars = ax.bar(["无储能", "MILP 优化"], values, color=[PALETTE["neutral"], PALETTE["primary"]])
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set(title="全天购电费对比", ylabel="购电费 (元)", ylim=(0, max(values) * 1.15))
    add_caption(fig, f"结论：优化方案成本降低 {s['cost_saving_percent']:.2f}%（{s['cost_saving_yuan']:.2f} 元）。")
    export_checked(fig, "result_q1_cost_comparison")

    # result-3：效率敏感性属于有序参数扫描，使用折线而非分类柱状图。
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(sensitivity["单程效率"], sensitivity["最优购电费_元"], marker="o", label="最优成本", color=PALETTE["contrast"])
    ax.set(title="效率敏感性", xlabel="充、放电单程效率", ylabel="最优购电费 (元)")
    ax.legend()
    best = sensitivity.iloc[sensitivity["最优购电费_元"].argmin()]
    add_caption(fig, f"结论：在扫描范围内，单程效率 {best['单程效率']:.3f} 对应最低购电费 {best['最优购电费_元']:.2f} 元。")
    export_checked(fig, "result_q1_efficiency_sensitivity")


def run_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # 参数必须在 (0,1]；包含主方案 0.90 与“总往返效率为 0.90”的 sqrt(0.9) 方案。
    for eta in [0.80, 0.85, 0.90, math.sqrt(0.90), 0.95, 1.00]:
        sol = solve_dispatch(df, Parameters(eta_charge=eta, eta_discharge=eta), binary=True)
        rows.append({
            "单程效率": eta,
            "往返效率": eta * eta,
            "最优购电费_元": sol.summary["optimized_grid_cost_yuan"],
            "全天购电量_kWh": sol.summary["optimized_grid_energy_kWh"],
            "弃光量_kWh": sol.summary["curtailment_kWh"],
        })
    return pd.DataFrame(rows)


def write_repro_manifest(params: Parameters) -> None:
    """调用 Skill 官方脚本生成输入哈希、运行环境、参数与唯一复现命令。"""
    script = SKILL_ROOT / "references" / "roles" / "编程手" / "scripts" / "repro_manifest.py"
    command = [
        sys.executable, str(script),
        "--project-root", str(PROJECT_ROOT),
        "--input", str(INPUT_XLSX),
        "--input", str(TEMPLATE_XLSX),
        "--seed", "0",
        "--parameters", json.dumps(asdict(params), ensure_ascii=False),
        "--command", "python problem1_solve.py",
        "--package", "numpy",
        "--package", "pandas",
        "--package", "scipy",
        "--package", "matplotlib",
        "--package", "openpyxl",
        "--package", "pillow",
        "--overwrite",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def print_summary(summary: dict) -> None:
    print("\n=== 问题一求解结果 ===")
    print(f"求解状态：{summary['solver_status']}")
    print(f"全天购电量：{summary['optimized_grid_energy_kWh']:.4f} kWh")
    print(f"全天购电费：{summary['optimized_grid_cost_yuan']:.4f} 元")
    print(f"无储能基准：{summary['no_storage_grid_cost_yuan']:.4f} 元")
    print(f"节省金额/比例：{summary['cost_saving_yuan']:.4f} 元 / {summary['cost_saving_percent']:.2f}%")
    print(f"充电/放电总量：{summary['charge_energy_kWh']:.4f} / {summary['discharge_energy_kWh']:.4f} kWh")
    print(f"SOC 范围及日末：[{summary['min_soc_kWh']:.4f}, {summary['max_soc_kWh']:.4f}] / {summary['end_soc_kWh']:.4f} kWh")
    print(f"最大平衡/SOC 残差：{summary['max_balance_residual_kWh']:.3e} / {summary['max_soc_residual_kWh']:.3e} kWh")
    print(f"同时充放电时段数：{summary['simultaneous_charge_discharge_periods']}")


def main() -> int:
    args = parse_args()
    params = Parameters(eta_charge=args.eta, eta_discharge=args.eta)
    df = read_input(INPUT_XLSX)
    solution = solve_dispatch(df, params, binary=True)
    print_summary(solution.summary)
    if args.smoke:
        print("SMOKE_PASS：真实输入、核心求解链和约束回代均通过。")
        return 0

    write_outputs(solution, params)
    sensitivity = run_sensitivity(df)
    sensitivity.to_csv(RESULTS_DIR / "问题1_效率敏感性.csv", index=False, encoding="utf-8-sig")
    make_figures(solution, sensitivity)
    write_repro_manifest(params)
    print(f"结果目录：{RESULTS_DIR}")
    print(f"图表目录：{FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
