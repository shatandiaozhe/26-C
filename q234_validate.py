#!/usr/bin/env python3
"""问题二至四：时间顺序五折检验、独立回代与百分之十扰动分析。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from q234_solve import (
    DynamicQuantileParameters, Parameters, _issue_index, dynamic_net_forecast,
    dynamic_validation_cost, forecast_load_median, load_inputs,
    official_pv_forecast, select_dynamic_quantile, validate_schedule,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "问题二至四模型检验"
RESULT_FILE = ROOT / "results" / "问题2至4_逐时段完整结果.csv"
SEED = 20260911
REPEATS = 100


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = pred - y
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    scale = float(np.mean(np.abs(y)))
    return {"RMSE_kWh": rmse, "MAE_kWh": mae,
            "nRMSE_百分比": 100 * rmse / scale if scale else 0.0}


def pinball(y: np.ndarray, pred: np.ndarray, quantile: float) -> float:
    error = y - pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def prediction_validation(data, p: Parameters) -> pd.DataFrame:
    """按日期连续分成五折；每一天的预测只调用此前历史。"""
    days = np.arange(31, 365)  # 一月仅作热启动，二月至十二月样本外评价
    folds = np.array_split(days, 5)
    cfg = DynamicQuantileParameters()
    history = {alpha: [] for alpha in cfg.candidates}
    selected_alpha = cfg.default_alpha
    dynamic_records = {}
    for day_i in range(365):
        selected_alpha, scores = select_dynamic_quantile(history, selected_alpha, cfg)
        dynamic_records[day_i] = (
            dynamic_net_forecast(data, day_i, selected_alpha, cfg),
            selected_alpha,
            scores.get(selected_alpha, np.nan),
        )
        for alpha in cfg.candidates:
            history[alpha].append(
                dynamic_validation_cost(data, day_i, alpha, cfg, p, "fixed")
            )
    rows = []
    issue_ts = [_issue_index(data.time_labels, h) for h in (0, 6, 12, 18)]
    for fold_no, fold in enumerate(folds, 1):
        actual_load, actual_pv, q2_net, actual_net = [], [], [], []
        selected_alphas, selected_scores = [], []
        q3_load, q3_pv = [], []
        naive_load, naive_pv = [], []
        for day_i in fold:
            actual_load.append(data.load_kwh[day_i]); actual_pv.append(data.pv_kwh[day_i])
            net_hat, alpha, score = dynamic_records[day_i]
            q2_net.append(net_hat)
            actual_net.append(data.load_kwh[day_i] - data.pv_kwh[day_i])
            selected_alphas.append(alpha); selected_scores.append(score)
            lag = day_i - 7
            naive_load.append(data.load_kwh[lag]); naive_pv.append(data.pv_kwh[lag])
            day_l, day_p = np.zeros(144), np.zeros(144)
            for k, start in enumerate(issue_ts):
                stop = issue_ts[k + 1] if k + 1 < len(issue_ts) else 144
                day_l[start:stop] = forecast_load_median(data, day_i, p, start)[start:stop]
                day_p[start:stop] = official_pv_forecast(data, day_i, start)[start:stop]
            q3_load.append(day_l); q3_pv.append(day_p)
        arrays = [np.concatenate(x) for x in
                  (actual_load, actual_pv, q2_net, actual_net, q3_load, q3_pv,
                   naive_load, naive_pv)]
        al, ap, n2, an, l3, p3, nl, npv = arrays
        naive_net = nl - npv
        m, b = metrics(an, n2), metrics(an, naive_net)
        rows.append({"折次": fold_no, "起始日期": str(data.dates[fold[0]].date()),
                     "结束日期": str(data.dates[fold[-1]].date()), "问题": "问题二及问题四日前分支",
                     "变量": "净负荷", **m, "周滞后基线RMSE_kWh": b["RMSE_kWh"],
                     "相对基线RMSE改善_百分比": 100 * (1 - m["RMSE_kWh"] / b["RMSE_kWh"]),
                     "平均选择分位数": float(np.mean(selected_alphas)),
                     "分位数最小值": float(np.min(selected_alphas)),
                     "分位数最大值": float(np.max(selected_alphas)),
                     "平均选择得分_元": float(np.mean(selected_scores)),
                     "Pinball损失": np.nan, "周滞后基线Pinball损失": np.nan,
                     "相对基线Pinball改善_百分比": np.nan})
        for question, variable, truth, pred, base, quantile in [
            ("问题三及问题四滚动分支", "负荷", al, l3, nl, .5),
            ("问题三及问题四滚动分支", "光伏", ap, p3, npv, .5),
        ]:
            m, b = metrics(truth, pred), metrics(truth, base)
            rows.append({"折次": fold_no, "起始日期": str(data.dates[fold[0]].date()),
                         "结束日期": str(data.dates[fold[-1]].date()), "问题": question,
                         "变量": variable, **m, "周滞后基线RMSE_kWh": b["RMSE_kWh"],
                         "相对基线RMSE改善_百分比": 100 * (1 - m["RMSE_kWh"] / b["RMSE_kWh"]),
                         "平均选择分位数": quantile, "分位数最小值": quantile,
                         "分位数最大值": quantile, "平均选择得分_元": np.nan,
                         "Pinball损失": pinball(truth, pred, quantile),
                         "周滞后基线Pinball损失": pinball(truth, base, quantile),
                         "相对基线Pinball改善_百分比": 100 * (1 - pinball(truth, pred, quantile)
                                                              / pinball(truth, base, quantile))})
    return pd.DataFrame(rows)


def replay_cost(frame: pd.DataFrame, load_factor: np.ndarray,
                pv_factor: np.ndarray, p: Parameters) -> dict[str, float]:
    load = frame["实际负荷_kWh"].to_numpy() * load_factor
    pv = frame["实际光伏_kWh"].to_numpy() * pv_factor
    balance = (frame["调整后购电量_kWh"].to_numpy() + pv
               + frame["放电量_kWh"].to_numpy() - load
               - frame["充电量_kWh"].to_numpy())
    emergency = np.maximum(-balance, 0.0)
    remaining = np.maximum(balance, 0.0)
    fixed = float((frame["计划购电费_元"] + frame["调整成本_元"]).sum())
    emergency_cost = float(np.sum(p.emergency_multiplier
                                  * frame["电价_元每kWh"].to_numpy() * emergency))
    return {"扰动后总成本_元": fixed + emergency_cost,
            "紧急购电量_kWh": float(emergency.sum()),
            "剩余电量_kWh": float(remaining.sum())}


def robustness(frame: pd.DataFrame, p: Parameters) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows = []
    official = frame[frame["日期"] >= "2025-02-01"].copy()
    for name, group in official.groupby("策略", sort=False):
        group = group.reset_index(drop=True)
        base = float(group["总成本_元"].sum())
        n = len(group)
        for kind in ("逐时段独立", "一小时相关块"):
            for repeat in range(1, REPEATS + 1):
                size = n if kind == "逐时段独立" else int(np.ceil(n / 6))
                nl = rng.uniform(-0.1, 0.1, size)
                npv = rng.uniform(-0.1, 0.1, size)
                if kind == "一小时相关块":
                    nl, npv = np.repeat(nl, 6)[:n], np.repeat(npv, 6)[:n]
                result = replay_cost(group, 1 + nl, 1 + npv, p)
                rows.append({"策略": name, "扰动类型": kind, "样本序号": repeat,
                             **result, "成本变化率_百分比":
                             100 * (result["扰动后总成本_元"] / base - 1)})
    detail = pd.DataFrame(rows)
    base_emergency = official.groupby("策略")["紧急购电量_kWh"].sum().to_dict()
    summaries = []
    for (name, kind), group in detail.groupby(["策略", "扰动类型"], sort=False):
        change = group["成本变化率_百分比"]
        summaries.append({"策略": name, "扰动类型": kind, "样本数": len(group),
            "成本变化均值_百分比": change.mean(), "成本变化标准差_百分点": change.std(ddof=1),
            "最大绝对成本变化_百分比": change.abs().max(),
            "成本变化百分位2_5": change.quantile(.025),
            "成本变化百分位97_5": change.quantile(.975),
            "紧急购电变化均值_百分比": 100 * (group["紧急购电量_kWh"].mean()
                / base_emergency[name] - 1)})
    summary = pd.DataFrame(summaries)
    return detail, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = Parameters()
    data = load_inputs(p)
    frame = pd.read_csv(RESULT_FILE, parse_dates=["日期"])
    checks = {name: validate_schedule(g, p) for name, g in frame.groupby("策略")}
    cv = prediction_validation(data, p)
    detail, robust = robustness(frame, p)
    cv.to_csv(OUT / "五折时间顺序预测检验.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT / "百分之十扰动逐情景结果.csv", index=False, encoding="utf-8-sig")
    robust.to_csv(OUT / "百分之十扰动汇总.csv", index=False, encoding="utf-8-sig")
    payload = {"随机种子": SEED, "每组情景数": REPEATS, "独立回代": checks,
               "五折平均": cv.groupby(["问题", "变量"]).mean(numeric_only=True).reset_index().to_dict("records"),
               "扰动汇总": robust.to_dict("records")}
    (OUT / "检验摘要.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
