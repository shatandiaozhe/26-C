#!/usr/bin/env python3
"""生成动态净负荷分位数方法 A-D 的同口径比较。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "动态分位数"
START = pd.Timestamp("2025-02-01")
CHINESE_FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")


def cvar95(values: pd.Series) -> float:
    threshold = values.quantile(0.95)
    return float(values[values >= threshold].mean())


baseline = pd.read_csv(ROOT / "results" / "问题2至4_逐时段完整结果.csv", parse_dates=["日期"])
bc = pd.read_csv(OUT / "方法B与方法C_逐时段结果.csv", parse_dates=["日期"])
d = pd.read_csv(OUT / "方法D_逐时段结果.csv", parse_dates=["日期"])
baseline = baseline[(baseline["策略"] == "问题2_固定价") & (baseline["日期"] >= START)].copy()
baseline["策略"] = "方法A_负荷0.80减光伏0.20"
all_rows = pd.concat([baseline, bc[bc["日期"] >= START], d[d["日期"] >= START]], ignore_index=True)

rows = []
for name, group in all_rows.groupby("策略", sort=False):
    daily = group.groupby("日期", as_index=False).agg(
        总成本_元=("总成本_元", "sum"),
        紧急购电量_kWh=("紧急购电量_kWh", "sum"),
        剩余电量_kWh=("剩余电量_kWh", "sum"),
        实际负荷_kWh=("实际负荷_kWh", "sum"),
        放电量_kWh=("放电量_kWh", "sum"),
    )
    rows.append({
        "方法": name,
        "总成本_元": float(daily["总成本_元"].sum()),
        "日成本CVaR95_元": cvar95(daily["总成本_元"]),
        "紧急购电量_kWh": float(daily["紧急购电量_kWh"].sum()),
        "紧急购电率": float(daily["紧急购电量_kWh"].sum() / daily["实际负荷_kWh"].sum()),
        "剩余电量_kWh": float(daily["剩余电量_kWh"].sum()),
        "按额定容量估算等效放电循环": float(daily["放电量_kWh"].sum() / 12000.0),
        "最大单日成本_元": float(daily["总成本_元"].max()),
    })
comparison = pd.DataFrame(rows).sort_values("总成本_元")
base_cost = float(comparison.loc[comparison["方法"].str.startswith("方法A"), "总成本_元"].iloc[0])
comparison["相对方法A成本变化"] = comparison["总成本_元"] / base_cost - 1.0

choices = pd.read_csv(OUT / "每日动态分位数.csv", parse_dates=["日期"])
choices = choices[choices["日期"] >= START]
alpha_counts = choices["选择分位数"].value_counts().sort_index()
selection = {
    "正式评价天数": int(len(choices)),
    "分位数均值": float(choices["选择分位数"].mean()),
    "分位数中位数": float(choices["选择分位数"].median()),
    "分位数变更次数": int((choices["选择分位数"].diff().abs() > 1e-12).sum()),
    "各分位数天数": {str(k): int(v) for k, v in alpha_counts.items()},
}

payload = {"方法比较": comparison.to_dict(orient="records"), "动态分位数": selection}
(OUT / "方法A-D比较.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# 动态净负荷分位数滚动回测",
    "",
    "正式评价区间为2025年2月1日至12月31日。所有方法只使用决策日以前的数据，方法C的候选分位数由最近14个已实现历史日的运行成本选择。",
    "",
    "| 方法 | 总成本/万元 | 相对A变化 | 日成本CVaR95/元 | 紧急购电率 | 剩余电量/万kWh | 等效放电循环 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for _, row in comparison.iterrows():
    lines.append(
        f"| {row['方法']} | {row['总成本_元']/1e4:.4f} | {row['相对方法A成本变化']:.2%} | "
        f"{row['日成本CVaR95_元']:.2f} | {row['紧急购电率']:.2%} | "
        f"{row['剩余电量_kWh']/1e4:.4f} | {row['按额定容量估算等效放电循环']:.2f} |"
    )
lines += [
    "",
    f"方法C正式评价期分位数均值为{selection['分位数均值']:.3f}，中位数为{selection['分位数中位数']:.2f}，共变更{selection['分位数变更次数']}次。",
    "",
    "方法D每小时重新优化后成本反而上升，原因是频繁调整需要按题目1.5倍上调和0.5倍下调规则结算。日内滚动不是天然更优，必须把调整费用纳入触发条件；后续应增加“预计节省超过调整成本才执行”的门槛。",
    "",
    "当前外层验证采用一小时聚合以控制嵌套混合整数求解规模，最终调度仍为10分钟粒度。现有数据没有天气、节假日和辐照度字段，因此相似日只使用星期类型、近期净负荷形态和时间衰减。",
]
(OUT / "动态净负荷分位数回测说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].bar(comparison["方法"].str.slice(0, 3), comparison["总成本_元"] / 1e4, color=["#009E73", "#56B4E9", "#E69F00", "#D55E00"])
axes[0].set_title("方法A-D总成本", fontproperties=CHINESE_FONT)
axes[0].set_ylabel("万元", fontproperties=CHINESE_FONT)
axes[0].set_xlabel("方法", fontproperties=CHINESE_FONT)
for label in axes[0].get_xticklabels():
    label.set_fontproperties(CHINESE_FONT)
axes[1].bar([f"{x:.2f}" for x in alpha_counts.index], alpha_counts.values, color="#0072B2")
axes[1].set_title("方法C每日选择分位数", fontproperties=CHINESE_FONT)
axes[1].set_ylabel("天数", fontproperties=CHINESE_FONT)
axes[1].set_xlabel("净负荷分位数", fontproperties=CHINESE_FONT)
fig.tight_layout()
fig.savefig(OUT / "方法A-D成本与分位数.png", dpi=220, bbox_inches="tight")
print(json.dumps(payload, ensure_ascii=False, indent=2))
