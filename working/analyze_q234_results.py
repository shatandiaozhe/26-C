"""问题二至问题四结果统计与敏感性复核。

该脚本读取正式求解结果，统一截取 2025-02-01 至 2025-12-31，
并在不覆盖原结果的前提下重算少量关键参数情景。
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import q234_solve as solver


RESULTS = ROOT / "results"
START = pd.Timestamp("2025-02-01")
CAPACITY_KWH = 12000.0
USABLE_CAPACITY_KWH = 9600.0
TOL = 1e-6


def scalar(value):
    """把 NumPy/Pandas 标量转换为可写入 JSON 的 Python 标量。"""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def describe(series: pd.Series) -> dict[str, float]:
    series = pd.to_numeric(series, errors="raise")
    return {
        "均值": float(series.mean()),
        "样本方差": float(series.var(ddof=1)),
        "标准差": float(series.std(ddof=1)),
        "最小值": float(series.min()),
        "中位数": float(series.median()),
        "最大值": float(series.max()),
        "95分位数": float(series.quantile(0.95)),
        "变异系数": float(series.std(ddof=1) / series.mean()) if series.mean() else None,
    }


def pct_change(new: float, base: float) -> float:
    return float((new - base) / base) if base else 0.0


def strategy_statistics(interval: pd.DataFrame, daily: pd.DataFrame) -> dict:
    name = str(interval["策略"].iloc[0])
    daily = daily.sort_values("日期").copy()
    emergency_intervals = interval[interval["紧急购电量_kWh"] > TOL]
    active_charge = interval[interval["充电量_kWh"] > TOL]
    active_discharge = interval[interval["放电量_kWh"] > TOL]
    soc_low = np.isclose(interval["期末SOC_kWh"], 1200.0, atol=1e-5)
    soc_high = np.isclose(interval["期末SOC_kWh"], 10800.0, atol=1e-5)
    max_emergency_row = interval.loc[interval["紧急购电量_kWh"].idxmax()]
    max_cost_row = daily.loc[daily["总成本_元"].idxmax()]
    min_cost_row = daily.loc[daily["总成本_元"].idxmin()]
    total_charge = float(interval["充电量_kWh"].sum())
    total_discharge = float(interval["放电量_kWh"].sum())
    total_cost = float(daily["总成本_元"].sum())
    total_emergency_cost = float(daily["紧急购电费_元"].sum())
    total_adjusted = float(daily["调整后购电量_kWh"].sum())
    total_emergency = float(daily["紧急购电量_kWh"].sum())
    totals = {col: float(daily[col].sum()) for col in daily.columns if col not in {"策略", "日期"}}
    return {
        "策略": name,
        "正式评价天数": int(daily["日期"].nunique()),
        "总量": totals,
        "日总成本描述": describe(daily["总成本_元"]),
        "日紧急购电描述": describe(daily["紧急购电量_kWh"]),
        "日剩余电量描述": describe(daily["剩余电量_kWh"]),
        "紧急购电": {
            "发生天数": int((daily["紧急购电量_kWh"] > TOL).sum()),
            "发生时段数": int(len(emergency_intervals)),
            "发生时段占比": float(len(emergency_intervals) / len(interval)),
            "紧急电量占实际外购电量": float(total_emergency / (total_adjusted + total_emergency)),
            "紧急费用占总成本": float(total_emergency_cost / total_cost),
            "最大单时段_kWh": float(max_emergency_row["紧急购电量_kWh"]),
            "最大单时段日期": max_emergency_row["日期"].isoformat(),
            "最大单时段时间": str(max_emergency_row["时间标签"]),
        },
        "极值日期": {
            "最高日成本日期": max_cost_row["日期"].isoformat(),
            "最高日成本_元": float(max_cost_row["总成本_元"]),
            "最低日成本日期": min_cost_row["日期"].isoformat(),
            "最低日成本_元": float(min_cost_row["总成本_元"]),
        },
        "储能运行": {
            "总充电量_kWh": total_charge,
            "总放电量_kWh": total_discharge,
            "充电时段数": int(len(active_charge)),
            "放电时段数": int(len(active_discharge)),
            "空闲时段数": int(((interval["充电量_kWh"] <= TOL) & (interval["放电量_kWh"] <= TOL)).sum()),
            "平均SOC_kWh": float(interval["期末SOC_kWh"].mean()),
            "SOC标准差_kWh": float(interval["期末SOC_kWh"].std(ddof=1)),
            "触及下界时段数": int(soc_low.sum()),
            "触及上界时段数": int(soc_high.sum()),
            "按额定容量估算等效放电循环": total_discharge / CAPACITY_KWH,
            "按可用容量估算等效放电循环": total_discharge / USABLE_CAPACITY_KWH,
        },
        "关联性": {
            "日紧急购电与日总成本相关系数": float(daily["紧急购电量_kWh"].corr(daily["总成本_元"])),
            "日剩余电量与日总成本相关系数": float(daily["剩余电量_kWh"].corr(daily["总成本_元"])),
            "时段购电量与电价相关系数": float(interval["调整后购电量_kWh"].corr(interval["电价_元每kWh"])),
        },
    }


def paired_comparison(daily: pd.DataFrame, base_name: str, new_name: str) -> dict:
    base = daily[daily["策略"] == base_name].set_index("日期")
    new = daily[daily["策略"] == new_name].set_index("日期")
    diff = base["总成本_元"] - new["总成本_元"]
    return {
        "基准策略": base_name,
        "比较策略": new_name,
        "总成本差_基准减比较_元": float(diff.sum()),
        "相对基准降幅": float(diff.sum() / base["总成本_元"].sum()),
        "逐日节省均值_元": float(diff.mean()),
        "逐日节省样本方差": float(diff.var(ddof=1)),
        "逐日节省标准差_元": float(diff.std(ddof=1)),
        "逐日节省中位数_元": float(diff.median()),
        "逐日节省最小值_元": float(diff.min()),
        "逐日节省最小值日期": diff.idxmin().isoformat(),
        "逐日节省最大值_元": float(diff.max()),
        "逐日节省最大值日期": diff.idxmax().isoformat(),
        "节省为正天数": int((diff > TOL).sum()),
        "节省为负天数": int((diff < -TOL).sum()),
        "成本相同天数": int((diff.abs() <= TOL).sum()),
    }


def summarize_scenario(frame: pd.DataFrame) -> dict[str, float]:
    official = frame[frame["日期"] >= START]
    return {
        "总成本_元": float(official["总成本_元"].sum()),
        "紧急购电量_kWh": float(official["紧急购电量_kWh"].sum()),
        "剩余电量_kWh": float(official["剩余电量_kWh"].sum()),
        "计划购电量_kWh": float(official["计划购电量_kWh"].sum()),
        "调整后购电量_kWh": float(official["调整后购电量_kWh"].sum()),
    }


def run_sensitivity() -> dict:
    baseline = solver.Parameters()
    cases = {
        "效率降低10%_eta0.81": replace(baseline, eta_charge=0.81, eta_discharge=0.81),
        "效率提高10%_eta0.99": replace(baseline, eta_charge=0.99, eta_discharge=0.99),
    }
    output: dict[str, dict] = {"储能效率": {}}
    for label, params in cases.items():
        solver.validate_parameters(params)
        data = solver.load_inputs(params)
        frames = {
            "问题2_固定价": solver.solve_day_ahead(data, params, "fixed", 365),
            "问题3_固定价": solver.solve_rolling(data, params, "fixed", 365),
            "问题4-2_波动价": solver.solve_day_ahead(data, params, "variable", 365),
            "问题4-3_波动价": solver.solve_rolling(data, params, "variable", 365),
        }
        output["储能效率"][label] = {name: summarize_scenario(frame) for name, frame in frames.items()}

    quantile_cases = {
        "较激进_qL0.72_qPV0.22": replace(baseline, load_quantile=0.72, pv_quantile=0.22),
        "基准_qL0.80_qPV0.20": baseline,
        "较保守_qL0.88_qPV0.18": replace(baseline, load_quantile=0.88, pv_quantile=0.18),
    }
    output["日前分位数"] = {}
    for label, params in quantile_cases.items():
        data = solver.load_inputs(params)
        frames = {
            "问题2_固定价": solver.solve_day_ahead(data, params, "fixed", 365),
            "问题4-2_波动价": solver.solve_day_ahead(data, params, "variable", 365),
        }
        output["日前分位数"][label] = {name: summarize_scenario(frame) for name, frame in frames.items()}
    return output


def main() -> None:
    interval = pd.read_csv(RESULTS / "问题2至4_逐时段完整结果.csv", parse_dates=["日期"])
    daily = pd.read_csv(RESULTS / "问题2至4_逐日汇总.csv", parse_dates=["日期"])
    voi = pd.read_csv(RESULTS / "问题3_新增预报时点_VOI.csv", parse_dates=["日期"])
    interval = interval[interval["日期"] >= START].copy()
    daily = daily[daily["日期"] >= START].copy()
    voi = voi[voi["日期"] >= START].copy()

    stats = {}
    for name, group in interval.groupby("策略", sort=False):
        day_group = daily[daily["策略"] == name]
        stats[name] = strategy_statistics(group, day_group)

    voi_base = daily[daily["策略"] == "问题3_固定价"].set_index("日期")
    voi_dense = voi.set_index("日期")
    voi_diff = voi_base["总成本_元"] - voi_dense["总成本_元"]

    price = interval.drop_duplicates(["日期", "时段序号"])["电价_元每kWh"]
    fixed_price = interval[interval["策略"] == "问题2_固定价"]["电价_元每kWh"]
    variable_price = interval[interval["策略"] == "问题4-2_波动价"]["电价_元每kWh"]
    physical = interval[interval["策略"] == "问题2_固定价"]
    representative_dates = pd.to_datetime(["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"])
    monthly = (
        daily.assign(月份=daily["日期"].dt.strftime("%Y-%m"))
        .groupby(["策略", "月份"], as_index=False)[["总成本_元", "紧急购电量_kWh", "剩余电量_kWh"]]
        .sum()
    )
    representative = daily[daily["日期"].isin(representative_dates)].copy()
    payload = {
        "统计口径": {
            "开始日期": START.isoformat(),
            "结束日期": daily["日期"].max().isoformat(),
            "正式评价天数": int(daily["日期"].nunique()),
            "每策略时段数": int(interval.groupby("策略").size().iloc[0]),
        },
        "实际供需总量": {
            "实际负荷_kWh": float(physical["实际负荷_kWh"].sum()),
            "实际光伏_kWh": float(physical["实际光伏_kWh"].sum()),
            "实际净负荷_kWh": float((physical["实际负荷_kWh"] - physical["实际光伏_kWh"]).sum()),
        },
        "策略统计": stats,
        "月度汇总": monthly.to_dict(orient="records"),
        "代表日汇总": representative.to_dict(orient="records"),
        "成对比较": {
            "固定价滚动相对日前": paired_comparison(daily, "问题2_固定价", "问题3_固定价"),
            "波动价滚动相对日前": paired_comparison(daily, "问题4-2_波动价", "问题4-3_波动价"),
            "波动价日前相对固定价日前": paired_comparison(daily, "问题2_固定价", "问题4-2_波动价"),
            "波动价滚动相对固定价滚动": paired_comparison(daily, "问题3_固定价", "问题4-3_波动价"),
        },
        "新增预报时点": {
            "四时点总成本_元": float(voi_base["总成本_元"].sum()),
            "八时点代理总成本_元": float(voi_dense["总成本_元"].sum()),
            "毛信息价值_元": float(voi_diff.sum()),
            "相对四时点降幅": float(voi_diff.sum() / voi_base["总成本_元"].sum()),
            "逐日毛节省描述": describe(voi_diff),
            "节省为正天数": int((voi_diff > TOL).sum()),
            "节省为负天数": int((voi_diff < -TOL).sum()),
            "节省为零天数": int((voi_diff.abs() <= TOL).sum()),
            "盈亏平衡日均新增系统成本_元": float(voi_diff.sum() / len(voi_diff)),
        },
        "电价分布": {
            "固定价": describe(fixed_price),
            "波动价": describe(variable_price),
        },
        "紧急电价倍数敏感性": {},
        "重算敏感性": run_sensitivity(),
        "基准参数": asdict(solver.Parameters()),
    }

    for multiplier in (4.5, 5.0, 5.5):
        label = f"{multiplier:.1f}倍"
        payload["紧急电价倍数敏感性"][label] = {}
        for name, item in stats.items():
            totals = item["总量"]
            non_emergency = totals["计划购电费_元"] + totals["调整成本_元"]
            emergency_energy_cost_at_base_price = totals["紧急购电费_元"] / 5.0
            cost = non_emergency + multiplier * emergency_energy_cost_at_base_price
            payload["紧急电价倍数敏感性"][label][name] = {
                "总成本_元": float(cost),
                "相对5倍变化": pct_change(cost, item["总量"]["总成本_元"]),
            }

    output = RESULTS / "问题2至4_结果分析统计.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=scalar), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
