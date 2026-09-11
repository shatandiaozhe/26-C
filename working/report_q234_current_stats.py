#!/usr/bin/env python3
"""从当前正式结果提取不触发敏感性重算的报告统计。"""

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "working"))
import analyze_q234_results as analysis

interval = pd.read_csv(ROOT / "results" / "问题2至4_逐时段完整结果.csv", parse_dates=["日期"])
daily = pd.read_csv(ROOT / "results" / "问题2至4_逐日汇总.csv", parse_dates=["日期"])
voi = pd.read_csv(ROOT / "results" / "问题3_新增预报时点_VOI.csv", parse_dates=["日期"])
interval = interval[interval["日期"] >= analysis.START]
daily = daily[daily["日期"] >= analysis.START]
voi = voi[voi["日期"] >= analysis.START]

payload = {
    "策略统计": {
        name: analysis.strategy_statistics(group, daily[daily["策略"] == name])
        for name, group in interval.groupby("策略", sort=False)
    },
    "固定价滚动相对日前": analysis.paired_comparison(daily, "问题2_固定价", "问题3_固定价"),
    "波动价滚动相对日前": analysis.paired_comparison(daily, "问题4-2_波动价", "问题4-3_波动价"),
}
base = daily[daily["策略"] == "问题3_固定价"].set_index("日期")["总成本_元"]
dense = voi.set_index("日期")["总成本_元"]
diff = base - dense
payload["新增预报时点"] = {
    "八时点总成本_元": float(dense.sum()),
    "毛信息价值_元": float(diff.sum()),
    "逐日": analysis.describe(diff),
    "节省为正天数": int((diff > analysis.TOL).sum()),
    "节省为负天数": int((diff < -analysis.TOL).sum()),
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
