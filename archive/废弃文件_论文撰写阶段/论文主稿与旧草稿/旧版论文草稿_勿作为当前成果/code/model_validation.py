"""
2020 C题模型检验与鲁棒性分析。

运行：python code/model_validation.py
依赖：numpy pandas openpyxl
输出：outputs/model_validation/
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import model_solver as ms

OUT = ROOT / "outputs" / "model_validation"
OUT.mkdir(parents=True, exist_ok=True)
SOL = ROOT / "outputs" / "model_solution"
RNG = np.random.default_rng(202008)


def kappa_binary(y, pred):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    po = np.mean(y == pred)
    py = np.array([np.mean(y == 0), np.mean(y == 1)])
    pp = np.array([np.mean(pred == 0), np.mean(pred == 1)])
    pe = float(py @ pp)
    return float((po - pe) / max(1 - pe, 1e-12))


def weighted_kappa(actual, predicted):
    labels = ["A", "B", "C", "D"]
    a = np.array([labels.index(v) for v in actual])
    p = np.array([labels.index(v) for v in predicted])
    n, k = len(a), 4
    o = np.zeros((k, k))
    for x, z in zip(a, p): o[x, z] += 1
    o /= n
    ha = np.bincount(a, minlength=k) / n
    hp = np.bincount(p, minlength=k) / n
    e = np.outer(ha, hp)
    w = np.fromfunction(lambda i, j: ((i-j)/(k-1))**2, (k, k))
    return float(1 - np.sum(w*o) / max(np.sum(w*e), 1e-12))


def rankdata(a):
    a = np.asarray(a)
    order = np.argsort(a)
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    for v in np.unique(a):
        mask = a == v
        if mask.sum() > 1: ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def ks_stat(y, p):
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    yy = y[order]
    c1 = np.cumsum(yy == 1) / max(np.sum(yy == 1), 1)
    c0 = np.cumsum(yy == 0) / max(np.sum(yy == 0), 1)
    return float(np.max(np.abs(c1-c0)))


def ece(y, p, bins=5):
    edges = np.linspace(0, 1, bins+1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            total += mask.mean() * abs(np.mean(p[mask]) - np.mean(y[mask]))
    return float(total)


def jaccard(a, b):
    a, b = set(np.where(np.asarray(a) > 0)[0]), set(np.where(np.asarray(b) > 0)[0])
    return len(a & b) / max(len(a | b), 1)


def normalized_reallocation(a, b, budget):
    # 除以2B：从一家转出1万元再转入另一家会产生2万元绝对变化。
    return float(np.sum(np.abs(np.asarray(a)-np.asarray(b))) / (2*budget))


def constraint_check(df, loan_col, grade_col, budget):
    loan = df[loan_col].to_numpy(float)
    positive = loan[loan > 0]
    return {
        "总额差": float(np.sum(loan)-budget),
        "低于10万元户数": int(np.sum((loan > 0) & (loan < 10))),
        "超过100万元户数": int(np.sum(loan > 100)),
        "D级贷款总额": float(df.loc[df[grade_col] == "D", loan_col].sum()),
        "约束通过": bool(abs(np.sum(loan)-budget)<1e-9 and np.all(positive>=10) and np.all(positive<=100) and df.loc[df[grade_col]=="D",loan_col].sum()==0),
    }


features, rate_table = ms.load_data()
source = features[features["样本类型"] == "有信贷记录"].copy().reset_index(drop=True)
target = features[features["样本类型"] == "无信贷记录"].copy().reset_index(drop=True)
xs = source[ms.RISK_FEATURES].to_numpy(float)
xt = target[ms.RISK_FEATURES].to_numpy(float)
y = source["是否违约"].eq("是").astype(int).to_numpy()
historical_grade = source["信誉评级"].astype(str).to_numpy()

q1 = pd.read_csv(SOL / "问题1_风险与信贷策略.csv")
q2 = pd.read_csv(SOL / "问题2_风险迁移与1亿元策略.csv")
q3 = pd.read_csv(SOL / "问题3_突发事件稳健调整策略.csv")
loss_scenarios = pd.read_csv(SOL / "问题3_MonteCarlo情景损失.csv")
model_summary = json.loads((SOL / "模型求解汇总.json").read_text(encoding="utf-8"))


# ---------- 问题1：重复分层五折交叉验证 ----------
safety, critic_w, _ = ms.critic_topsis(xs, ms.DIRECTION)
topsis_risk = 1-safety
cv_rows = []
all_oof = []
for repeat in range(20):
    pred_l, pred_f, theta, m = ms.cross_validated_logistic(xs, y, topsis_risk)
    # 更换折分随机种子：复用底层函数逻辑但显式重建折。
    folds = ms.stratified_folds(y, ms.PARAM["cv_folds"], 2020+repeat)
    pred_l = np.zeros(len(y))
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        xtr, xte, _, _ = ms.safe_standardize(xs[train], xs[test])
        beta = ms.fit_logistic_irls(xtr, y[train], ms.PARAM["ridge"])
        pred_l[test] = ms.predict_logistic(xte, beta)
    grid = np.linspace(0,1,101)
    theta = float(grid[np.argmin([np.mean((t*pred_l+(1-t)*topsis_risk-y)**2) for t in grid])])
    pred_f = theta*pred_l+(1-theta)*topsis_risk
    pred_cls = (pred_f >= .5).astype(int)
    met = ms.classification_metrics(y, pred_f)
    cv_rows.append({
        "重复": repeat+1, "AUC": met["AUC"], "RMSE": float(np.sqrt(np.mean((pred_f-y)**2))),
        "Brier": met["Brier"], "Recall": met["Recall"], "Specificity": met["Specificity"],
        "Kappa": kappa_binary(y,pred_cls), "KS": ks_stat(y,pred_f), "ECE": ece(y,pred_f), "theta": theta,
    })
    all_oof.append(pred_f)
cv = pd.DataFrame(cv_rows)
cv.to_csv(OUT / "问题1_重复五折交叉验证.csv", index=False, encoding="utf-8-sig")

mean_oof = np.mean(np.vstack(all_oof), axis=0)
cuts1 = ms.grade_thresholds(mean_oof, source["信誉评级"])
pred_grade1 = ms.assign_grade(mean_oof, cuts1)
rating_consistency = {
    "预测风险等级与人工评级加权Kappa": weighted_kappa(historical_grade, pred_grade1),
    "TOPSIS安全度与人工评级序数Spearman": spearman(safety, -np.array([{"A":1,"B":2,"C":3,"D":4}[g] for g in historical_grade])),
}


# ---------- 问题2：域对齐与迁移稳定性 ----------
all_raw = np.vstack([xs,xt])
all_z, _, _ = ms.safe_standardize(all_raw)
xs0, xt0 = all_z[:len(xs)], all_z[len(xs):]
aligned0, cov_before, cov_after = ms.coral_transform(xs0,xt0)

def smd_matrix(a,b):
    den = np.sqrt((np.var(a,axis=0)+np.var(b,axis=0))/2)
    den[den<1e-12]=1
    return np.abs(np.mean(a,axis=0)-np.mean(b,axis=0))/den

smd_before = smd_matrix(xs0,xt0)
smd_after = smd_matrix(aligned0,xt0)
domain_test = {
    "协方差差异_对齐前": float(cov_before), "协方差差异_对齐后": float(cov_after),
    "协方差改善率": float(1-cov_after/max(cov_before,1e-12)),
    "平均SMD_对齐前": float(np.mean(smd_before)), "平均SMD_对齐后": float(np.mean(smd_after)),
    "最大SMD_对齐前": float(np.max(smd_before)), "最大SMD_对齐后": float(np.max(smd_after)),
}


# ---------- 输入噪声鲁棒性 ----------
robust_rows = []
base_risk1 = q1.set_index("企业代号").loc[source["企业代号"],"融合违约风险"].to_numpy(float)
base_loan1 = q1.set_index("企业代号").loc[source["企业代号"],"贷款额度_万元"].to_numpy(float)
base_grade1 = source["信誉评级"].astype(str).to_numpy()
theta_base = float(model_summary["问题1"]["融合系数theta"])

# 问题1：扰动特征后完整重训。
for noise in [0.05,0.10]:
    for rep in range(20):
        xn = xs*(1+RNG.normal(0,noise,xs.shape))
        safe_n,_,_=ms.critic_topsis(xn,ms.DIRECTION)
        xz,_,_=ms.safe_standardize(xn)
        beta=ms.fit_logistic_irls(xz,y,ms.PARAM["ridge"])
        risk=np.clip(theta_base*ms.predict_logistic(xz,beta)+(1-theta_base)*(1-safe_n),.001,.999)
        alloc=ms.allocate_loans(risk,base_grade1,np.zeros(len(source)),rate_table,5000,force_full=True)
        robust_rows.append({"小问":"问题1","噪声比例":noise,"重复":rep+1,"风险排名Spearman":spearman(base_risk1,risk),
                            "评级一致率":np.nan,"获贷集合Jaccard":jaccard(base_loan1,alloc["loan"]),
                            "预算重分配比例":normalized_reallocation(base_loan1,alloc["loan"],5000),
                            "CVaR相对变化":np.nan})

# 问题2：固定Bootstrap索引，比较目标域扰动前后预测。
boot_rng=np.random.default_rng(888)
boot_idx=[]
for _ in range(40):
    idx=boot_rng.integers(0,len(source),len(source))
    while len(np.unique(y[idx]))<2: idx=boot_rng.integers(0,len(source),len(source))
    boot_idx.append(idx)

def q2_predict(target_raw):
    combined=np.vstack([xs,target_raw])
    z,_,_=ms.safe_standardize(combined)
    xsa, xta=z[:len(xs)],z[len(xs):]
    aligned,_,_=ms.coral_transform(xsa,xta)
    bp=[]
    for idx in boot_idx:
        beta=ms.fit_logistic_irls(aligned[idx],y[idx],ms.PARAM["ridge"])
        bp.append(ms.predict_logistic(xta,beta))
    bp=np.vstack(bp)
    risk=np.clip(bp.mean(axis=0)+ms.PARAM["uncertainty_eta"]*bp.std(axis=0,ddof=1),.001,.999)
    cuts=tuple(model_summary["问题2"]["评级阈值"])
    grade=ms.assign_grade(risk,cuts)
    alloc=ms.allocate_loans(risk,grade,bp.std(axis=0,ddof=1),rate_table,10000,force_full=True)
    return risk,grade,alloc["loan"]

base_r2,base_g2,base_l2=q2_predict(xt)
for noise in [0.05,0.10]:
    for rep in range(20):
        xtn=xt*(1+RNG.normal(0,noise,xt.shape))
        r,g,l=q2_predict(xtn)
        robust_rows.append({"小问":"问题2","噪声比例":noise,"重复":rep+1,"风险排名Spearman":spearman(base_r2,r),
                            "评级一致率":float(np.mean(base_g2==g)),"获贷集合Jaccard":jaccard(base_l2,l),
                            "预算重分配比例":normalized_reallocation(base_l2,l,10000),"CVaR相对变化":np.nan})

# 问题3：对冲击前风险增加噪声，重新模拟和优化；每次固定情景随机种子。
target["行业"]=target["企业名称"].map(ms.infer_industry)
base_pre=q2.set_index("企业代号").loc[target["企业代号"],"修正违约风险"].to_numpy(float)
std2=q2.set_index("企业代号").loc[target["企业代号"],"预测不确定性"].to_numpy(float)
cuts=tuple(model_summary["问题2"]["评级阈值"])

def q3_solve(pre_risk):
    ms.RNG=np.random.default_rng(24680)
    shocks,sp,_=ms.simulate_shocks(target,pre_risk)
    rr=np.clip(.55*sp.mean(axis=0)+.45*np.quantile(sp,.95,axis=0),.001,.999)
    gg=ms.assign_grade(rr,cuts)
    aa=ms.allocate_loans(rr,gg,std2,rate_table,10000,risk_aversion=.55,force_full=True)
    losses=np.sum(aa["loan"][None,:]*(1-aa["churn"])[None,:]*sp*ms.PARAM["lgd"],axis=1)
    _,cv=ms.cvar(losses,.95)
    return rr,gg,aa["loan"],cv

br3,bg3,bl3,bcv3=q3_solve(base_pre)
for noise in [0.05,0.10]:
    for rep in range(20):
        noisy=np.clip(base_pre*(1+RNG.normal(0,noise,len(base_pre))),.001,.999)
        r,g,l,cv3=q3_solve(noisy)
        robust_rows.append({"小问":"问题3","噪声比例":noise,"重复":rep+1,"风险排名Spearman":spearman(br3,r),
                            "评级一致率":float(np.mean(bg3==g)),"获贷集合Jaccard":jaccard(bl3,l),
                            "预算重分配比例":normalized_reallocation(bl3,l,10000),
                            "CVaR相对变化":float((cv3-bcv3)/bcv3)})

robust=pd.DataFrame(robust_rows)
robust.to_csv(OUT/"输入噪声鲁棒性明细.csv",index=False,encoding="utf-8-sig")
robust_summary=robust.groupby(["小问","噪声比例"]).agg(
    风险排名Spearman均值=("风险排名Spearman","mean"),风险排名Spearman标准差=("风险排名Spearman","std"),
    评级一致率均值=("评级一致率","mean"),获贷集合Jaccard均值=("获贷集合Jaccard","mean"),
    预算重分配比例均值=("预算重分配比例","mean"),预算重分配比例最大值=("预算重分配比例","max"),
    CVaR相对变化均值=("CVaR相对变化","mean"),CVaR相对变化最大绝对值=("CVaR相对变化",lambda x:np.nanmax(np.abs(x)) if x.notna().any() else np.nan)
).reset_index()
robust_summary.to_csv(OUT/"输入噪声鲁棒性汇总.csv",index=False,encoding="utf-8-sig")


# ---------- 问题3 Monte Carlo收敛 ----------
loss=loss_scenarios["问题3调整方案损失_万元"].to_numpy(float)
conv=[]
_,final_cv=ms.cvar(loss,.95)
for n in [200,400,600,800,1000,1200]:
    va,cvv=ms.cvar(loss[:n],.95)
    conv.append({"情景数":n,"均值损失":float(np.mean(loss[:n])),"VaR95":va,"CVaR95":cvv,"CVaR相对最终偏差":float((cvv-final_cv)/final_cv)})
pd.DataFrame(conv).to_csv(OUT/"问题3_MonteCarlo收敛.csv",index=False,encoding="utf-8-sig")


# ---------- 优化约束与尾部改善 ----------
constraints={
    "问题1":constraint_check(q1,"贷款额度_万元","信誉评级",5000),
    "问题2":constraint_check(q2,"贷款额度_万元","预测信誉评级",10000),
    "问题3":constraint_check(q3,"调整后额度_万元","调整后评级",10000),
}
base_loss=loss_scenarios["问题2原方案损失_万元"].to_numpy(float)
_,base_cvar=ms.cvar(base_loss,.95)
tail_test={
    "原方案平均损失":float(np.mean(base_loss)),"调整方案平均损失":float(np.mean(loss)),
    "平均损失降低率":float(1-np.mean(loss)/np.mean(base_loss)),
    "原方案CVaR95":base_cvar,"调整方案CVaR95":final_cv,"CVaR降低率":float(1-final_cv/base_cvar),
}

cv_summary=cv.drop(columns="重复").agg(["mean","std","min","max"]).T.reset_index().rename(columns={"index":"指标"})
cv_summary.to_csv(OUT/"问题1_交叉验证汇总.csv",index=False,encoding="utf-8-sig")

result={
    "问题1重复交叉验证":cv_summary.to_dict("records"),
    "问题1评级一致性":rating_consistency,
    "问题2域适应检验":domain_test,
    "问题3尾部风险检验":tail_test,
    "优化约束检验":constraints,
    "10%噪声摘要":robust_summary[robust_summary["噪声比例"]==.10].to_dict("records"),
    "MonteCarlo收敛":conv,
}
(OUT/"模型检验汇总.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")

print(json.dumps(result,ensure_ascii=False,indent=2))
