#!/usr/bin/env python3
"""问题二动态净负荷分位数：严格时间顺序的外层选参与内层 MILP 调度。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import q234_solve as base


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "动态分位数"


@dataclass(frozen=True)
class SelectionParameters:
    training_days: int = 60
    validation_days: int = 14
    similar_days: int = 20
    candidates: tuple[float, ...] = tuple(np.round(np.arange(0.50, 0.951, 0.05), 2))
    cvar_level: float = 0.95
    risk_weight: float = 0.20
    stability_weight: float = 0.02
    default_alpha: float = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="滚动选择净负荷风险分位数")
    parser.add_argument("--smoke", action="store_true", help="仅计算前80天")
    parser.add_argument("--hourly-only", action="store_true", help="读取已选分位数，只计算方法D小时滚动策略")
    return parser.parse_args()


def similar_days(data: base.InputData, day_i: int, cfg: SelectionParameters) -> np.ndarray:
    """仅在过去训练窗内选同星期优先、近期净负荷水平相近的历史日。"""
    start = max(0, day_i - cfg.training_days)
    candidates = np.arange(start, day_i, dtype=int)
    if not len(candidates):
        return candidates
    weekday = data.dates[day_i].weekday()
    same_type = candidates[data.dates[candidates].weekday == weekday]
    pool = same_type if len(same_type) >= min(5, cfg.similar_days) else candidates
    recent_start = max(0, day_i - 7)
    reference = np.mean(data.load_kwh[recent_start:day_i] - data.pv_kwh[recent_start:day_i], axis=0)
    net = data.load_kwh[pool] - data.pv_kwh[pool]
    scale = max(float(np.std(net)), 1.0)
    shape_distance = np.sqrt(np.mean(((net - reference) / scale) ** 2, axis=1))
    recency = (day_i - pool) / max(cfg.training_days, 1)
    score = shape_distance + 0.15 * recency
    return pool[np.argsort(score)[: cfg.similar_days]]


def net_quantile_forecast(
    data: base.InputData,
    day_i: int,
    alpha: float,
    cfg: SelectionParameters,
) -> np.ndarray:
    indices = similar_days(data, day_i, cfg)
    if not len(indices):
        return data.fallback_load_kwh - data.fallback_pv_kwh
    paired_net = data.load_kwh[indices] - data.pv_kwh[indices]
    return np.quantile(paired_net, alpha, axis=0)


def _hourly(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(24, 6).sum(axis=1)


def validation_operating_cost(
    data: base.InputData,
    day_i: int,
    alpha: float,
    cfg: SelectionParameters,
    params: base.Parameters,
) -> float:
    """以一小时粒度模拟历史日，保持日初日末 SOC 相同以公平比较候选分位数。"""
    net_hat = _hourly(net_quantile_forecast(data, day_i, alpha, cfg))
    load_hat = np.maximum(net_hat, 0.0)
    pv_hat = np.maximum(-net_hat, 0.0)
    price = data.fixed_price.reshape(24, 6).mean(axis=1)
    hourly_params = base.Parameters(
        dt_hours=1.0,
        max_power_kw=params.max_power_kw,
        eta_charge=params.eta_charge,
        eta_discharge=params.eta_discharge,
    )
    plan = base._dispatch_milp(load_hat, pv_hat, price, 6000.0, 6000.0, hourly_params)
    actual_net = _hourly(data.load_kwh[day_i] - data.pv_kwh[day_i])
    balance = plan.grid + plan.discharge - plan.charge - actual_net
    emergency = np.maximum(-balance, 0.0)
    return float(np.sum(price * plan.grid + params.emergency_multiplier * price * emergency))


def empirical_cvar(costs: np.ndarray, level: float) -> float:
    """验证样本较少时，取至少一个最差样本的平均值。"""
    count = max(1, int(np.ceil((1.0 - level) * len(costs))))
    return float(np.mean(np.sort(costs)[-count:]))


def select_alpha(
    history: dict[float, list[float]],
    previous_alpha: float,
    cfg: SelectionParameters,
) -> tuple[float, dict[float, float]]:
    scores: dict[float, float] = {}
    for alpha in cfg.candidates:
        costs = np.asarray(history[alpha][-cfg.validation_days :], dtype=float)
        if len(costs) < cfg.validation_days:
            scores[alpha] = np.inf
            continue
        mean_cost = float(np.mean(costs))
        risk = empirical_cvar(costs, cfg.cvar_level)
        stability = cfg.stability_weight * mean_cost * abs(alpha - previous_alpha) / 0.05
        scores[alpha] = mean_cost + cfg.risk_weight * risk + stability
    finite = {a: s for a, s in scores.items() if np.isfinite(s)}
    return (min(finite, key=finite.get) if finite else cfg.default_alpha), scores


def run_strategy(
    data: base.InputData,
    params: base.Parameters,
    cfg: SelectionParameters,
    stop_day: int,
    dynamic: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy = "方法C_滚动净负荷分位数" if dynamic else "方法B_固定净负荷0.80分位数"
    history = {alpha: [] for alpha in cfg.candidates}
    selected = cfg.default_alpha
    soc = params.soc_initial_kwh
    frames: list[pd.DataFrame] = []
    choices: list[dict] = []
    for day_i in range(stop_day):
        if dynamic:
            selected, scores = select_alpha(history, selected, cfg)
        else:
            selected, scores = cfg.default_alpha, {}
        net_hat = net_quantile_forecast(data, day_i, selected, cfg)
        price = data.fixed_price
        terminal = params.soc_initial_kwh if day_i == stop_day - 1 else None
        plan = base._dispatch_milp(
            np.maximum(net_hat, 0.0), np.maximum(-net_hat, 0.0), price,
            soc, terminal, params,
        )
        executed = base.execute_segment(
            data, day_i, 0, 144, plan.grid, plan.charge, plan.discharge, soc, params
        )
        frames.append(base._daily_records(
            data, day_i, price, plan.grid, plan.grid, plan.charge, plan.discharge,
            executed["soc"], executed["emergency"], executed["spill"], soc, params, strategy,
        ))
        choices.append({
            "日期": data.dates[day_i],
            "选择分位数": selected,
            "验证样本数": min(day_i, cfg.validation_days),
            "选择得分": scores.get(selected, np.nan),
        })
        soc = float(executed["end_soc"])
        # 当天结束后才把当天反事实成本加入历史，确保选参不使用待决策日实际值。
        if dynamic:
            for alpha in cfg.candidates:
                history[alpha].append(validation_operating_cost(data, day_i, alpha, cfg, params))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(choices)


def run_hourly_strategy(
    data: base.InputData,
    params: base.Parameters,
    cfg: SelectionParameters,
    stop_day: int,
    choices: pd.DataFrame,
) -> pd.DataFrame:
    """方法D：沿用当天因果选择的分位数，每小时校正净负荷并只执行下一小时。"""
    alpha_by_date = {
        pd.Timestamp(row["日期"]).normalize(): float(row["选择分位数"])
        for _, row in choices.iterrows()
    }
    frames: list[pd.DataFrame] = []
    soc = params.soc_initial_kwh
    for day_i in range(stop_day):
        alpha = alpha_by_date[data.dates[day_i].normalize()]
        initial_forecast = net_quantile_forecast(data, day_i, alpha, cfg)
        terminal = params.soc_initial_kwh if day_i == stop_day - 1 else None
        plan0 = base._dispatch_milp(
            np.maximum(initial_forecast, 0.0), np.maximum(-initial_forecast, 0.0),
            data.fixed_price, soc, terminal, params,
        )
        q0 = plan0.grid.copy()
        q_final = q0.copy()
        charge_final = plan0.charge.copy()
        discharge_final = plan0.discharge.copy()
        soc_final = np.zeros(144)
        emergency_final = np.zeros(144)
        spill_final = np.zeros(144)
        day_start_soc = soc
        for start in range(0, 144, 6):
            stop = min(start + 6, 144)
            if start == 0:
                current = plan0
            else:
                forecast = net_quantile_forecast(data, day_i, alpha, cfg)
                recent_start = max(0, start - 12)
                actual_net = data.load_kwh[day_i] - data.pv_kwh[day_i]
                bias = float(np.mean(actual_net[recent_start:start] - forecast[recent_start:start]))
                steps = np.arange(144 - start)
                updated = forecast[start:] + bias * np.exp(-steps / 36.0)
                current = base._dispatch_milp(
                    np.maximum(updated, 0.0), np.maximum(-updated, 0.0),
                    data.fixed_price[start:], soc, terminal, params, base_grid=q0[start:],
                )
                q_final[start:] = current.grid
                charge_final[start:] = current.charge
                discharge_final[start:] = current.discharge
            segment = base.execute_segment(
                data, day_i, start, stop,
                current.grid[: stop - start], current.charge[: stop - start],
                current.discharge[: stop - start], soc, params,
            )
            q_final[start:stop] = current.grid[: stop - start]
            charge_final[start:stop] = current.charge[: stop - start]
            discharge_final[start:stop] = current.discharge[: stop - start]
            soc_final[start:stop] = segment["soc"]
            emergency_final[start:stop] = segment["emergency"]
            spill_final[start:stop] = segment["spill"]
            soc = float(segment["end_soc"])
        frames.append(base._daily_records(
            data, day_i, data.fixed_price, q0, q_final, charge_final, discharge_final,
            soc_final, emergency_final, spill_final, day_start_soc, params,
            "方法D_动态分位数加小时滚动",
        ))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    params = base.Parameters()
    cfg = SelectionParameters()
    base.validate_parameters(params)
    data = base.load_inputs(params)
    stop_day = 80 if args.smoke else 365

    if args.hourly_only:
        choices = pd.read_csv(RESULTS / "每日动态分位数.csv", parse_dates=["日期"])
        hourly = run_hourly_strategy(data, params, cfg, stop_day, choices)
        check = base.validate_schedule(hourly, params)
        RESULTS.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(RESULTS / "方法D_逐时段结果.csv", index=False, encoding="utf-8-sig")
        base.summarize(hourly).to_csv(RESULTS / "方法D_逐日汇总.csv", index=False, encoding="utf-8-sig")
        official = hourly[hourly["日期"] >= pd.Timestamp("2025-02-01")]
        print(json.dumps({"总成本_元": float(official["总成本_元"].sum()), "校验": check}, ensure_ascii=False, indent=2))
        return

    fixed, _ = run_strategy(data, params, cfg, stop_day, dynamic=False)
    dynamic, choices = run_strategy(data, params, cfg, stop_day, dynamic=True)
    checks = {
        name: base.validate_schedule(group, params)
        for name, group in pd.concat([fixed, dynamic]).groupby("策略")
    }
    if args.smoke:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        print(choices.tail(10).to_string(index=False))
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    all_rows = pd.concat([fixed, dynamic], ignore_index=True)
    all_rows.to_csv(RESULTS / "方法B与方法C_逐时段结果.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(RESULTS / "每日动态分位数.csv", index=False, encoding="utf-8-sig")
    daily = base.summarize(all_rows)
    daily.to_csv(RESULTS / "方法B与方法C_逐日汇总.csv", index=False, encoding="utf-8-sig")
    official = daily[daily["日期"] >= pd.Timestamp("2025-02-01")]
    totals = official.groupby("策略")["总成本_元"].sum().to_dict()
    payload = {"配置": asdict(cfg), "总成本_元": totals, "校验": checks}
    (RESULTS / "动态分位数摘要.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
