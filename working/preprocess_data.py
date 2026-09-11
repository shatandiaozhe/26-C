from __future__ import annotations

import json
import math
import re
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "附件"
OUT = PROJECT_ROOT / "working" / "prepared_data"
OUT.mkdir(parents=True, exist_ok=True)


def json_ready(frame: pd.DataFrame) -> list[dict]:
    cleaned = frame.copy()
    for col in cleaned.columns:
        if pd.api.types.is_datetime64_any_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    cleaned = cleaned.astype(object).where(pd.notna(cleaned), None)
    return cleaned.to_dict(orient="records")


def dump_json(name: str, obj) -> None:
    (OUT / name).write_text(
        json.dumps(obj, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def time_label(value, slot: int | None = None) -> str:
    if slot == 144 or "+1" in str(value):
        return "24:00"
    if isinstance(value, time):
        return f"{value.hour:02d}:{value.minute:02d}"
    text = str(value).strip()
    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if match:
        return f"{int(match.group(1)):02d}:{int(match.group(2)):02d}"
    return text


def issue_hour(value) -> int:
    if isinstance(value, time):
        return value.hour
    return int(str(value).strip().split(":")[0])


def wide_to_long(path: Path, sheet_name: str, value_name: str) -> pd.DataFrame:
    wide = pd.read_excel(path, sheet_name=sheet_name)
    date_col = wide.columns[0]
    wide[date_col] = pd.to_datetime(wide[date_col])
    time_cols = list(wide.columns[1:])
    column_to_slot = {col: idx + 1 for idx, col in enumerate(time_cols)}
    long = wide.melt(id_vars=[date_col], var_name="原时刻", value_name=value_name)
    long["时段序号"] = long["原时刻"].map(column_to_slot).astype(int)
    long["日期"] = pd.to_datetime(long[date_col]).dt.normalize()
    long["日期时间"] = long["日期"] + pd.to_timedelta(long["时段序号"] * 10, unit="m")
    long["时刻"] = [time_label(v, s) for v, s in zip(long["原时刻"], long["时段序号"])]
    return long[["日期时间", "日期", "时刻", "时段序号", value_name]].sort_values(
        ["日期", "时段序号"]
    )


def contextual_iqr(frame: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)[value_col]
    q1 = grouped.transform(lambda s: s.quantile(0.25))
    q3 = grouped.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    flag = (iqr > 0) & ((frame[value_col] < lower) | (frame[value_col] > upper))
    return pd.DataFrame({"下界": lower, "上界": upper, "四分位距": iqr, "异常": flag})


def isolated_iqr(series: pd.Series) -> pd.DataFrame:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    flag = (series < lower) | (series > upper)
    return pd.DataFrame(
        {
            "下界": np.repeat(lower, len(series)),
            "上界": np.repeat(upper, len(series)),
            "四分位距": np.repeat(iqr, len(series)),
            "异常": flag.fillna(False),
        }
    )


def descriptive_row(dataset: str, variable: str, values: pd.Series, flags: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce")
    return {
        "数据集": dataset,
        "变量": variable,
        "样本数": int(clean.notna().sum()),
        "缺失数": int(clean.isna().sum()),
        "最小值": float(clean.min()),
        "下四分位数": float(clean.quantile(0.25)),
        "中位数": float(clean.median()),
        "上四分位数": float(clean.quantile(0.75)),
        "最大值": float(clean.max()),
        "均值": float(clean.mean()),
        "标准差": float(clean.std(ddof=0)),
        "偏度": float(clean.skew()),
        "四分位距异常数": int(flags.sum()),
        "异常率": float(flags.mean()),
        "负值数": int((clean < 0).sum()),
        "处理动作": "无异常，无需处理" if int(flags.sum()) == 0 else "保留并标记",
    }


def histogram_rows(dataset: str, variable: str, values: pd.Series, bins: int = 24) -> list[dict]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    counts, edges = np.histogram(clean, bins=bins)
    rows = []
    for left, right, count in zip(edges[:-1], edges[1:], counts):
        rows.append(
            {
                "数据集": dataset,
                "变量": variable,
                "区间": f"{left:.2f}—{right:.2f}",
                "区间中点": float((left + right) / 2),
                "频数": int(count),
            }
        )
    return rows


def main() -> None:
    descriptions: list[dict] = []
    anomaly_details: list[dict] = []
    histogram: list[dict] = []

    # 附件1：单日数据。
    a1 = pd.read_excel(ROOT / "附件1.xlsx")
    a1.columns = ["原时刻", "电价", "小区负载功率", "光伏预测功率"]
    a1["时段序号"] = np.arange(1, len(a1) + 1)
    a1["时刻"] = [time_label(v, s) for v, s in zip(a1["原时刻"], a1["时段序号"])]
    a1 = a1[["时段序号", "时刻", "电价", "小区负载功率", "光伏预测功率"]]
    a1["负荷电量"] = a1["小区负载功率"] / 6
    a1["光伏预测电量"] = a1["光伏预测功率"] / 6
    for col in ["电价", "小区负载功率", "光伏预测功率"]:
        audit = isolated_iqr(a1[col])
        flag_col = f"{col}异常"
        a1[flag_col] = audit["异常"].astype(bool)
        descriptions.append(descriptive_row("附件1", col, a1[col], a1[flag_col]))
        histogram.extend(histogram_rows("附件1", col, a1[col]))
        for idx in a1.index[a1[flag_col]]:
            anomaly_details.append(
                {
                    "数据集": "附件1",
                    "变量": col,
                    "日期时间或定位": a1.loc[idx, "时刻"],
                    "数值": float(a1.loc[idx, col]),
                    "检测下界": float(audit.loc[idx, "下界"]),
                    "检测上界": float(audit.loc[idx, "上界"]),
                    "检测规则": "单日全局四分位距",
                    "处理动作": "保留并标记",
                    "理由": "未发现物理非法值，单日尖峰可能是真实负荷、电价或光伏变化",
                }
            )

    # 附件2和附件4：转换为统一的10分钟长表。
    load = wide_to_long(ROOT / "附件2.xlsx", "小区负载", "小区负载功率")
    pv = wide_to_long(ROOT / "附件2.xlsx", "光伏发电实际功率", "光伏实际功率")
    price = wide_to_long(ROOT / "附件4.xlsx", "Sheet1", "波动电价")
    actual = load.merge(
        pv[["日期", "时段序号", "光伏实际功率"]], on=["日期", "时段序号"], how="left"
    ).merge(
        price[["日期", "时段序号", "波动电价"]], on=["日期", "时段序号"], how="left"
    )
    actual["月份"] = actual["日期"].dt.month
    actual["星期"] = actual["日期"].dt.dayofweek + 1
    actual["年内日序"] = actual["日期"].dt.dayofyear
    actual["负荷电量"] = actual["小区负载功率"] / 6
    actual["光伏实际电量"] = actual["光伏实际功率"] / 6
    actual["时刻正弦"] = np.sin(2 * math.pi * actual["时段序号"] / 144)
    actual["时刻余弦"] = np.cos(2 * math.pi * actual["时段序号"] / 144)
    actual["年内正弦"] = np.sin(2 * math.pi * actual["年内日序"] / 365)
    actual["年内余弦"] = np.cos(2 * math.pi * actual["年内日序"] / 365)

    annual_variables = ["小区负载功率", "光伏实际功率", "波动电价"]
    audit_cache: dict[str, pd.DataFrame] = {}
    for col in annual_variables:
        audit = contextual_iqr(actual, col, ["月份", "时段序号"])
        physical_invalid = actual[col] < 0
        flag_col = f"{col}异常"
        actual[flag_col] = (audit["异常"] | physical_invalid).astype(bool)
        audit_cache[col] = audit
        descriptions.append(descriptive_row("全年实际数据", col, actual[col], actual[flag_col]))
        histogram.extend(histogram_rows("全年实际数据", col, actual[col]))
        for idx in actual.index[actual[flag_col]]:
            anomaly_details.append(
                {
                    "数据集": "全年实际数据",
                    "变量": col,
                    "日期时间或定位": actual.loc[idx, "日期时间"].strftime("%Y-%m-%d %H:%M"),
                    "数值": float(actual.loc[idx, col]),
                    "检测下界": float(audit.loc[idx, "下界"]),
                    "检测上界": float(audit.loc[idx, "上界"]),
                    "检测规则": "同月同一时刻四分位距",
                    "处理动作": "保留并标记",
                    "理由": "数值非负且时间索引完整，统计异常可能代表真实运行峰值",
                }
            )

    # 使用1月份作为初始训练窗口生成无未来信息泄露的标准化参数。
    january = actual[actual["月份"] == 1]
    standardization = []
    for col in annual_variables:
        mean = float(january[col].mean())
        std = float(january[col].std(ddof=0))
        min_value = float(january[col].min())
        max_value = float(january[col].max())
        standardization.append(
            {
                "变量": col,
                "拟合区间": "2025-01-01至2025-01-31",
                "样本数": int(january[col].notna().sum()),
                "均值": mean,
                "标准差": std,
                "最小值": min_value,
                "最大值": max_value,
                "标准化用途": "仅供近邻、支持向量等距离型预测模型；优化模型保留原单位",
            }
        )
        actual[f"{col}标准分数"] = (actual[col] - mean) / std if std else np.nan

    # 附件3：日期列的空白是合并单元格产生的结构性缺失。
    a3_raw = pd.read_excel(ROOT / "附件3.xlsx")
    a3_raw.columns = ["日期", "预报时刻"] + [f"预报{i}小时" for i in range(1, 25)]
    date_missing_original = a3_raw["日期"].isna()
    a3_raw["日期结构性填充"] = date_missing_original
    a3_raw["日期"] = pd.to_datetime(a3_raw["日期"].ffill())
    forecast_cols = [f"预报{i}小时" for i in range(1, 25)]
    forecast = a3_raw.melt(
        id_vars=["日期", "预报时刻", "日期结构性填充"],
        value_vars=forecast_cols,
        var_name="预测步长字段",
        value_name="光伏预测功率",
    )
    forecast["预测步长小时"] = forecast["预测步长字段"].str.extract(r"(\d+)").astype(int)
    forecast["发布小时"] = forecast["预报时刻"].map(issue_hour)
    forecast["发布时间"] = forecast["日期"] + pd.to_timedelta(forecast["发布小时"], unit="h")
    forecast["目标时间"] = forecast["发布时间"] + pd.to_timedelta(
        forecast["预测步长小时"], unit="h"
    )
    forecast["月份"] = forecast["日期"].dt.month
    forecast = forecast.sort_values(["发布时间", "预测步长小时"]).reset_index(drop=True)
    f_audit = contextual_iqr(forecast, "光伏预测功率", ["月份", "发布小时", "预测步长小时"])
    forecast["光伏预测功率异常"] = (f_audit["异常"] | (forecast["光伏预测功率"] < 0)).astype(bool)

    pv_map = actual[["日期时间", "光伏实际功率"]].drop_duplicates("日期时间")
    forecast = forecast.merge(pv_map, left_on="目标时间", right_on="日期时间", how="left")
    forecast["预测误差"] = forecast["光伏预测功率"] - forecast["光伏实际功率"]
    forecast["绝对误差"] = forecast["预测误差"].abs()
    forecast = forecast.drop(columns=["日期时间", "预测步长字段"])
    descriptions.append(
        descriptive_row(
            "附件3光伏预测", "光伏预测功率", forecast["光伏预测功率"], forecast["光伏预测功率异常"]
        )
    )
    histogram.extend(histogram_rows("附件3光伏预测", "光伏预测功率", forecast["光伏预测功率"]))
    for idx in forecast.index[forecast["光伏预测功率异常"]]:
        anomaly_details.append(
            {
                "数据集": "附件3光伏预测",
                "变量": "光伏预测功率",
                "日期时间或定位": (
                    f"发布{forecast.loc[idx, '发布时间']:%Y-%m-%d %H:%M}，"
                    f"预测{forecast.loc[idx, '预测步长小时']}小时"
                ),
                "数值": float(forecast.loc[idx, "光伏预测功率"]),
                "检测下界": float(f_audit.loc[idx, "下界"]),
                "检测上界": float(f_audit.loc[idx, "上界"]),
                "检测规则": "同月同发布时间同预测步长四分位距",
                "处理动作": "保留并标记",
                "理由": "预测值非负，极端预测可能反映天气变化，不直接删除",
            }
        )

    monthly = []
    for month in range(1, 13):
        subset = actual[actual["月份"] == month]
        for col in annual_variables:
            flag_col = f"{col}异常"
            monthly.append(
                {
                    "月份": month,
                    "变量": col,
                    "样本数": int(len(subset)),
                    "异常数": int(subset[flag_col].sum()),
                    "异常率": float(subset[flag_col].mean()),
                }
            )

    missing_summary = [
        {
            "数据集": "附件1",
            "字段": "全部字段",
            "原始缺失数": int(a1.isna().sum().sum()),
            "缺失性质": "无缺失",
            "处理方法": "不填充",
            "处理后缺失数": int(a1.isna().sum().sum()),
        },
        {
            "数据集": "附件2",
            "字段": "小区负载和光伏实际功率",
            "原始缺失数": int(load["小区负载功率"].isna().sum() + pv["光伏实际功率"].isna().sum()),
            "缺失性质": "无缺失",
            "处理方法": "不填充",
            "处理后缺失数": 0,
        },
        {
            "数据集": "附件3",
            "字段": "日期",
            "原始缺失数": int(date_missing_original.sum()),
            "缺失性质": "合并单元格导致的结构性空白",
            "处理方法": "按同一日期的四个预报时刻向下填充",
            "处理后缺失数": int(a3_raw["日期"].isna().sum()),
        },
        {
            "数据集": "附件3",
            "字段": "预报时刻和24小时光伏预测值",
            "原始缺失数": int(a3_raw[["预报时刻"] + forecast_cols].isna().sum().sum()),
            "缺失性质": "无缺失",
            "处理方法": "不填充",
            "处理后缺失数": int(a3_raw[["预报时刻"] + forecast_cols].isna().sum().sum()),
        },
        {
            "数据集": "附件4",
            "字段": "波动电价",
            "原始缺失数": int(price["波动电价"].isna().sum()),
            "缺失性质": "无缺失",
            "处理方法": "不填充",
            "处理后缺失数": int(price["波动电价"].isna().sum()),
        },
        {
            "数据集": "附件3与附件2匹配",
            "字段": "预测目标时间对应的实际光伏",
            "原始缺失数": int(forecast["光伏实际功率"].isna().sum()),
            "缺失性质": "年末预测超出附件2实际数据范围",
            "处理方法": "保留为空，不外推实际值，不参与误差评价",
            "处理后缺失数": int(forecast["光伏实际功率"].isna().sum()),
        },
    ]

    quality_checks = [
        {"检查项": "附件1行数", "结果": len(a1), "预期": 144, "结论": "通过" if len(a1) == 144 else "异常"},
        {
            "检查项": "附件2全年10分钟记录数",
            "结果": len(actual),
            "预期": 365 * 144,
            "结论": "通过" if len(actual) == 365 * 144 else "异常",
        },
        {
            "检查项": "全年日期—时段键重复数",
            "结果": int(actual.duplicated(["日期", "时段序号"]).sum()),
            "预期": 0,
            "结论": "通过" if not actual.duplicated(["日期", "时段序号"]).any() else "异常",
        },
        {
            "检查项": "附件3预报发布记录数",
            "结果": len(a3_raw),
            "预期": 365 * 4,
            "结论": "通过" if len(a3_raw) == 365 * 4 else "异常",
        },
        {
            "检查项": "附件3长表预测记录数",
            "结果": len(forecast),
            "预期": 365 * 4 * 24,
            "结论": "通过" if len(forecast) == 365 * 4 * 24 else "异常",
        },
        {
            "检查项": "负功率或负价格记录数",
            "结果": int(sum((actual[col] < 0).sum() for col in annual_variables) + (forecast["光伏预测功率"] < 0).sum()),
            "预期": 0,
            "结论": "通过" if not any((actual[col] < 0).any() for col in annual_variables) and not (forecast["光伏预测功率"] < 0).any() else "需核验",
        },
    ]

    a1_export = a1.rename(
        columns={
            "小区负载功率": "小区负载功率_千瓦",
            "光伏预测功率": "光伏预测功率_千瓦",
            "负荷电量": "负荷电量_千瓦时",
            "光伏预测电量": "光伏预测电量_千瓦时",
        }
    )
    actual_export = actual.rename(
        columns={
            "小区负载功率": "小区负载功率_千瓦",
            "光伏实际功率": "光伏实际功率_千瓦",
            "波动电价": "波动电价_元每千瓦时",
            "负荷电量": "负荷电量_千瓦时",
            "光伏实际电量": "光伏实际电量_千瓦时",
            "小区负载功率标准分数": "负荷标准分数_一月参数",
            "光伏实际功率标准分数": "光伏标准分数_一月参数",
            "波动电价标准分数": "电价标准分数_一月参数",
        }
    )
    forecast_export = forecast.rename(
        columns={
            "光伏预测功率": "光伏预测功率_千瓦",
            "光伏实际功率": "光伏实际功率_千瓦",
            "预测误差": "预测误差_千瓦",
            "绝对误差": "绝对误差_千瓦",
        }
    )

    dump_json("附件1处理后.json", json_ready(a1_export))
    dump_json("全年实际处理后.json", json_ready(actual_export))
    dump_json("光伏预测处理后.json", json_ready(forecast_export))
    dump_json("描述统计.json", descriptions)
    dump_json("缺失值汇总.json", missing_summary)
    dump_json("异常值明细.json", anomaly_details)
    dump_json("月度异常汇总.json", monthly)
    dump_json("直方图数据.json", histogram)
    dump_json("标准化参数.json", standardization)
    dump_json("质量检查.json", quality_checks)

    compact = {
        "附件1记录数": len(a1),
        "全年实际记录数": len(actual),
        "光伏预测记录数": len(forecast),
        "结构性日期填充数": int(date_missing_original.sum()),
        "年末无对应实际光伏数": int(forecast["光伏实际功率"].isna().sum()),
        "异常值明细数": len(anomaly_details),
        "描述统计": descriptions,
        "质量检查": quality_checks,
    }
    dump_json("审计摘要.json", compact)
    print("预处理完成")


if __name__ == "__main__":
    main()
