#!/usr/bin/env python3
"""独立核对问题二至四跨日储能衔接和全年末端状态。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "results" / "问题2至4_逐时段完整结果.csv", parse_dates=["日期"])
audit = {}
for name, group in data.groupby("策略"):
    group = group.sort_values(["日期", "时段序号"])
    daily = group.groupby("日期").agg(
        start=("期初SOC_kWh", "first"),
        end=("期末SOC_kWh", "last"),
    )
    audit[name] = {
        "非末日首末不同天数": int(((daily.iloc[:-1].end - daily.iloc[:-1].start).abs() > 1e-6).sum()),
        "最大跨日衔接残差_kWh": float(
            np.max(np.abs(daily.start.iloc[1:].to_numpy() - daily.end.iloc[:-1].to_numpy()))
        ),
        "初始SOC_kWh": float(daily.start.iloc[0]),
        "最终SOC_kWh": float(daily.end.iloc[-1]),
    }

print(json.dumps(audit, ensure_ascii=False, indent=2))
