"""对 model_solver.py 的三问结果做描述统计、关联分析和参数敏感性重算。"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import model_solver as ms

OUT = ROOT / "outputs" / "result_analysis"
OUT.mkdir(parents=True, exist_ok=True)
SOL = ROOT / "outputs" / "model_solution"


def desc(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return {
        "样本数": int(s.size), "均值": float(s.mean()), "标准差": float(s.std(ddof=1)),
        "方差": float(s.var(ddof=1)), "最小值": float(s.min()), "Q1": float(s.quantile(.25)),
        "中位数": float(s.median()), "Q3": float(s.quantile(.75)), "最大值": float(s.max()),
    }


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


q1 = pd.read_csv(SOL / "问题1_风险与信贷策略.csv")
q2 = pd.read_csv(SOL / "问题2_风险迁移与1亿元策略.csv")
q3 = pd.read_csv(SOL / "问题3_突发事件稳健调整策略.csv")
sc = pd.read_csv(SOL / "问题3_MonteCarlo情景损失.csv")
summary = json.loads((SOL / "模型求解汇总.json").read_text(encoding="utf-8"))

analysis = {"问题1": {}, "问题2": {}, "问题3": {}}

# 问题1：整体、获贷企业、真实违约组之间的风险差异。
analysis["问题1"]["融合风险描述"] = desc(q1["融合违约风险"])
analysis["问题1"]["获贷额度描述"] = desc(q1.loc[q1["贷款额度_万元"] > 0, "贷款额度_万元"])
analysis["问题1"]["利率描述"] = desc(q1.loc[q1["贷款额度_万元"] > 0, "贷款年利率"])
analysis["问题1"]["客户流失率描述"] = desc(q1.loc[q1["贷款额度_万元"] > 0, "客户流失率"])
analysis["问题1"]["相关性"] = {
    "风险与额度": corr(q1["融合违约风险"], q1["贷款额度_万元"]),
    "风险与利率": corr(q1["融合违约风险"], q1["贷款年利率"]),
    "风险与预期损失": corr(q1["融合违约风险"], q1["预期损失_万元"]),
}
analysis["问题1"]["违约组风险均值"] = q1.groupby("是否违约")["融合违约风险"].mean().to_dict()
analysis["问题1"]["评级汇总"] = q1.groupby("信誉评级").agg(
    企业数=("企业代号", "count"), 风险均值=("融合违约风险", "mean"),
    贷款总额=("贷款额度_万元", "sum"), 平均额度=("贷款额度_万元", "mean"),
    预期净收益=("预期净收益_万元", "sum"), 预期损失=("预期损失_万元", "sum")
).reset_index().to_dict("records")

# 问题2：预测风险、置信度、评级和贷款的关系。
analysis["问题2"]["修正风险描述"] = desc(q2["修正违约风险"])
analysis["问题2"]["不确定性描述"] = desc(q2["预测不确定性"])
analysis["问题2"]["获贷额度描述"] = desc(q2.loc[q2["贷款额度_万元"] > 0, "贷款额度_万元"])
analysis["问题2"]["相关性"] = {
    "风险与额度": corr(q2["修正违约风险"], q2["贷款额度_万元"]),
    "风险与利率": corr(q2["修正违约风险"], q2["贷款年利率"]),
    "不确定性与额度": corr(q2["预测不确定性"], q2["贷款额度_万元"]),
    "风险与预期损失": corr(q2["修正违约风险"], q2["预期损失_万元"]),
}
analysis["问题2"]["评级汇总"] = q2.groupby("预测信誉评级").agg(
    企业数=("企业代号", "count"), 风险均值=("修正违约风险", "mean"),
    风险标准差=("修正违约风险", "std"), 贷款总额=("贷款额度_万元", "sum"),
    平均额度=("贷款额度_万元", "mean"), 平均利率=("贷款年利率", "mean"),
    预期净收益=("预期净收益_万元", "sum"), 预期损失=("预期损失_万元", "sum")
).reset_index().to_dict("records")

# 问题3：风险上升、额度变化和情景损失尾部。
q3["风险增量"] = q3["冲击后平均风险"] - q3["冲击前风险"]
analysis["问题3"]["平均冲击描述"] = desc(q3["平均冲击强度"])
analysis["问题3"]["风险增量描述"] = desc(q3["风险增量"])
analysis["问题3"]["额度变化描述"] = desc(q3["额度变化_万元"])
analysis["问题3"]["调整后损失描述"] = desc(sc["问题3调整方案损失_万元"])
analysis["问题3"]["原方案损失描述"] = desc(sc["问题2原方案损失_万元"])
analysis["问题3"]["相关性"] = {
    "冲击强度与风险增量": corr(q3["平均冲击强度"], q3["风险增量"]),
    "风险增量与额度变化": corr(q3["风险增量"], q3["额度变化_万元"]),
    "企业暴露度与风险增量": corr(q3["企业暴露度"], q3["风险增量"]),
}
analysis["问题3"]["行业汇总"] = q3.groupby("行业").agg(
    企业数=("企业代号", "count"), 平均冲击=("平均冲击强度", "mean"),
    风险平均增量=("风险增量", "mean"), 原额度=("问题2额度_万元", "sum"),
    调整后额度=("调整后额度_万元", "sum"), 净调整=("额度变化_万元", "sum")
).reset_index().to_dict("records")


# 重新构造敏感性分析所需输入。
features, rate_table = ms.load_data()
source = features[features["样本类型"] == "有信贷记录"].copy().reset_index(drop=True)
target = features[features["样本类型"] == "无信贷记录"].copy().reset_index(drop=True)
risk1 = q1.set_index("企业代号").loc[source["企业代号"], "融合违约风险"].to_numpy(float)
grade1 = source["信誉评级"].astype(str).to_numpy()
risk2_mean = q2.set_index("企业代号").loc[target["企业代号"], "CORAL平均违约风险"].to_numpy(float)
risk2_std = q2.set_index("企业代号").loc[target["企业代号"], "预测不确定性"].to_numpy(float)
base_eta = ms.PARAM["uncertainty_eta"]
base_lgd = ms.PARAM["lgd"]
base_lam = ms.PARAM["risk_aversion"]
base_rho = ms.PARAM["propagation_rho"]
base_gamma = ms.PARAM["shock_gamma"]
cuts = tuple(summary["问题2"]["评级阈值"])

sensitivity = []

# 问题1：LGD ±10%。
for lgd in [base_lgd * .9, base_lgd, base_lgd * 1.1]:
    ms.PARAM["lgd"] = lgd
    a = ms.allocate_loans(risk1, grade1, np.zeros(len(source)), rate_table, 5000, force_full=True)
    sensitivity.append(["问题1", "LGD", lgd, int(np.sum(a["loan"] > 0)), int(np.sum(a["loan"])), float(np.sum(a["net"])), float(np.sum(a["expected_loss"])), np.nan])
ms.PARAM["lgd"] = base_lgd

# 问题1：风险厌恶系数 ±10%。
for lam in [base_lam * .9, base_lam, base_lam * 1.1]:
    a = ms.allocate_loans(risk1, grade1, np.zeros(len(source)), rate_table, 5000, risk_aversion=lam, force_full=True)
    sensitivity.append(["问题1", "风险厌恶系数", lam, int(np.sum(a["loan"] > 0)), int(np.sum(a["loan"])), float(np.sum(a["net"])), float(np.sum(a["expected_loss"])), np.nan])

# 问题2：不确定性惩罚系数 ±10%，同时重算修正风险、等级与配置。
for eta in [base_eta * .9, base_eta, base_eta * 1.1]:
    ms.PARAM["uncertainty_eta"] = eta
    r = np.clip(risk2_mean + eta * risk2_std, .001, .999)
    g = ms.assign_grade(r, cuts)
    a = ms.allocate_loans(r, g, risk2_std, rate_table, 10000, force_full=True)
    sensitivity.append(["问题2", "不确定性惩罚eta", eta, int(np.sum(a["loan"] > 0)), int(np.sum(a["loan"])), float(np.sum(a["net"])), float(np.sum(a["expected_loss"])), np.nan])
ms.PARAM["uncertainty_eta"] = base_eta

# 问题3：传播系数rho ±10%，使用相同随机种子保证只比较参数变化。
target["行业"] = target["企业名称"].map(ms.infer_industry)
for rho in [base_rho * .9, base_rho, base_rho * 1.1]:
    ms.PARAM["propagation_rho"] = rho
    ms.PARAM["shock_gamma"] = base_gamma
    ms.RNG = np.random.default_rng(ms.SEED)
    shocks, stress_prob, exposure = ms.simulate_shocks(target, np.clip(risk2_mean + base_eta*risk2_std, .001, .999))
    robust = np.clip(.55*np.mean(stress_prob, axis=0)+.45*np.quantile(stress_prob,.95,axis=0),.001,.999)
    g = ms.assign_grade(robust, cuts)
    a = ms.allocate_loans(robust, g, risk2_std, rate_table, 10000, risk_aversion=.55, force_full=True)
    losses = np.sum(a["loan"][None,:]*(1-a["churn"])[None,:]*stress_prob*base_lgd, axis=1)
    _, cv = ms.cvar(losses,.95)
    sensitivity.append(["问题3", "传播系数rho", rho, int(np.sum(a["loan"] > 0)), int(np.sum(a["loan"])), np.nan, float(np.mean(losses)), cv])

# 问题3：风险放大系数gamma ±10%。
ms.PARAM["propagation_rho"] = base_rho
for gamma in [base_gamma * .9, base_gamma, base_gamma * 1.1]:
    ms.PARAM["shock_gamma"] = gamma
    ms.RNG = np.random.default_rng(ms.SEED)
    shocks, stress_prob, exposure = ms.simulate_shocks(target, np.clip(risk2_mean + base_eta*risk2_std, .001, .999))
    robust = np.clip(.55*np.mean(stress_prob, axis=0)+.45*np.quantile(stress_prob,.95,axis=0),.001,.999)
    g = ms.assign_grade(robust, cuts)
    a = ms.allocate_loans(robust, g, risk2_std, rate_table, 10000, risk_aversion=.55, force_full=True)
    losses = np.sum(a["loan"][None,:]*(1-a["churn"])[None,:]*stress_prob*base_lgd, axis=1)
    _, cv = ms.cvar(losses,.95)
    sensitivity.append(["问题3", "风险放大gamma", gamma, int(np.sum(a["loan"] > 0)), int(np.sum(a["loan"])), np.nan, float(np.mean(losses)), cv])

ms.PARAM["lgd"], ms.PARAM["risk_aversion"], ms.PARAM["uncertainty_eta"] = base_lgd, base_lam, base_eta
ms.PARAM["propagation_rho"], ms.PARAM["shock_gamma"] = base_rho, base_gamma

sens = pd.DataFrame(sensitivity, columns=["小问", "参数", "参数值", "获贷企业数", "贷款总额_万元", "预期净收益_万元", "预期或平均损失_万元", "CVaR95_万元"])
sens.to_csv(OUT / "参数敏感性分析.csv", index=False, encoding="utf-8-sig")

pd.DataFrame(analysis["问题1"]["评级汇总"]).to_csv(OUT / "问题1_分级统计.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(analysis["问题2"]["评级汇总"]).to_csv(OUT / "问题2_分级统计.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(analysis["问题3"]["行业汇总"]).to_csv(OUT / "问题3_行业统计.csv", index=False, encoding="utf-8-sig")
(OUT / "详细统计结果.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps(analysis, ensure_ascii=False, indent=2))
print("\nSENSITIVITY\n", sens.to_string(index=False))
