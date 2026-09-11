"""
2020 C题：中小微企业信贷决策——三问一体化求解程序

运行：python model_solver.py
核心依赖：numpy, pandas, openpyxl
输出：outputs/model_solution/ 下的 CSV、JSON、Markdown 和 SVG 图表。

设计说明：
1. 不依赖 scipy / sklearn / matplotlib，适合零基础队伍直接运行；
2. Logistic、AUC、分层交叉验证、CORAL 均用 NumPy 实现；
3. 优化采用“离散利率枚举 + 风险调整边际收益贪心配置”，每1万元为一步，
   能严格满足单户10—100万元与总预算上限；
4. 随机过程统一固定种子，保证每次运行结果一致。
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import traceback

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "outputs" / "data_stage" / "2020C_数据预处理与特征库.xlsx"
RATE_FILE = ROOT / "C" / "附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx"
OUT = ROOT / "outputs" / "model_solution"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2020
RNG = np.random.default_rng(SEED)

# 参数全部集中在这里，方便零基础队伍修改和敏感性分析。
PARAM = {
    "question1_budget_wan": 5000,   # 问题1题目未给具体总额，以5000万元作基准情景
    "question2_budget_wan": 10000,  # 1亿元=10000万元
    "min_loan_wan": 10,
    "max_loan_wan": 100,
    "lgd": 0.60,                    # 违约损失率；必须在[0,1]
    "risk_aversion": 0.35,          # 风险惩罚系数；越大越保守
    "uncertainty_eta": 0.20,        # 问题2不确定性惩罚；建议0~0.5
    "coral_eps": 1e-4,
    "ridge": 0.80,                  # Logistic L2正则，防止123个小样本过拟合
    "cv_folds": 5,
    "bootstrap_times": 80,
    "mc_scenarios": 1200,
    "cvar_alpha": 0.95,             # 必须在(0,1)
    "shock_gamma": 1.15,
    "propagation_rho": 0.30,        # 必须在[0,1]
}


RISK_FEATURES = [
    "进项_有效绝对额", "销项_有效绝对额", "净流入代理",
    "进项_交易对手数", "销项_交易对手数",
    "综合作废率", "综合负数率", "上下游集中度",
    "经营波动率", "经营趋势",
    "进项_最大交易对手占比", "销项_最大交易对手占比",
]

# 1表示数值越大越安全，-1表示数值越大风险越高。
DIRECTION = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1], dtype=float)


def validate_parameters() -> None:
    """尽早发现参数错误，避免程序运行很久后才报错。"""
    for key in ["lgd", "propagation_rho"]:
        if not 0 <= PARAM[key] <= 1:
            raise ValueError(f"参数 {key} 必须位于[0,1]")
    if not 0 < PARAM["cvar_alpha"] < 1:
        raise ValueError("cvar_alpha 必须位于(0,1)")
    if PARAM["min_loan_wan"] <= 0 or PARAM["max_loan_wan"] < PARAM["min_loan_wan"]:
        raise ValueError("贷款额度上下限设置错误")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """数据输入：优先读取上一阶段输出的统一工作簿。"""
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"未找到特征库：{FEATURE_FILE}")
    features = pd.read_excel(FEATURE_FILE, sheet_name="缩尾特征", engine="openpyxl")
    # 读取附件3的双层表头。iloc[2:]跳过标题行和评级行。
    raw = pd.read_excel(RATE_FILE, sheet_name="Sheet1", header=None, engine="openpyxl")
    rate = raw.iloc[2:31, :4].copy()
    rate.columns = ["rate", "A", "B", "C"]
    rate = rate.apply(pd.to_numeric, errors="coerce").dropna(subset=["rate"])
    return features, rate


def safe_standardize(train: np.ndarray, other: np.ndarray | None = None):
    """用训练集均值和标准差变换，避免把测试集信息泄漏进训练过程。"""
    mean = np.mean(train, axis=0)  # np.mean语义明确，也避免手写sum/n时的类型问题
    std = np.std(train, axis=0)
    std[std < 1e-12] = 1.0
    a = (train - mean) / std
    if other is None:
        return a, mean, std
    return a, (other - mean) / std, mean, std


def minmax_benefit(x: np.ndarray, direction: np.ndarray):
    """统一正负向指标，输出越大越安全的[0,1]矩阵。"""
    lo, hi = np.min(x, axis=0), np.max(x, axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    z = (x - lo) / span
    z[:, direction < 0] = 1.0 - z[:, direction < 0]
    return z, lo, hi


def critic_topsis(x: np.ndarray, direction: np.ndarray):
    """CRITIC客观赋权 + TOPSIS低风险接近度。"""
    z, lo, hi = minmax_benefit(x, direction)
    sigma = np.std(z, axis=0)
    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    information = sigma * np.sum(1.0 - corr, axis=1)
    weights = information / max(np.sum(information), 1e-12)
    v = z * weights
    ideal, worst = np.max(v, axis=0), np.min(v, axis=0)
    d_plus = np.sqrt(np.sum((v - ideal) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((v - worst) ** 2, axis=1))
    safety = d_minus / np.maximum(d_plus + d_minus, 1e-12)
    return safety, weights, (lo, hi)


def sigmoid(z):
    z = np.clip(z, -35, 35)  # 防止exp溢出
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_irls(x: np.ndarray, y: np.ndarray, ridge=0.8, max_iter=100):
    """类别加权Logistic的IRLS/Newton求解；无需sklearn。"""
    n, p = x.shape
    xd = np.column_stack([np.ones(n), x])
    beta = np.zeros(p + 1)
    n1, n0 = max(np.sum(y == 1), 1), max(np.sum(y == 0), 1)
    class_w = np.where(y == 1, n / (2 * n1), n / (2 * n0))
    penalty = np.eye(p + 1) * ridge
    penalty[0, 0] = 0.0  # 截距不正则化，避免整体概率被无意义压向0.5
    for _ in range(max_iter):
        prob = sigmoid(xd @ beta)
        w = class_w * np.maximum(prob * (1 - prob), 1e-6)
        grad = xd.T @ (class_w * (y - prob)) - penalty @ beta
        hess = xd.T @ (xd * w[:, None]) + penalty
        step = np.linalg.solve(hess + 1e-8 * np.eye(p + 1), grad)
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < 1e-7:
            beta = beta_new
            break
        beta = beta_new
    return beta


def predict_logistic(x: np.ndarray, beta: np.ndarray):
    return sigmoid(np.column_stack([np.ones(len(x)), x]) @ beta)


def stratified_folds(y: np.ndarray, k=5, seed=2020):
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(k)]
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for j, val in enumerate(idx):
            parts[j % k].append(int(val))
    return [np.array(sorted(v), dtype=int) for v in parts]


def auc_score(y: np.ndarray, p: np.ndarray):
    """用秩和公式计算AUC；0.5相当于随机判断，1为完美区分。"""
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # 对相同预测概率使用平均秩。
    for val in np.unique(p):
        mask = p == val
        if np.sum(mask) > 1:
            ranks[mask] = np.mean(ranks[mask])
    n1, n0 = np.sum(y == 1), np.sum(y == 0)
    return (np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def classification_metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return {
        "AUC": float(auc_score(y, p)),
        "Brier": float(np.mean((p - y) ** 2)),
        "Recall": tp / max(tp + fn, 1),
        "Specificity": tn / max(tn + fp, 1),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def cross_validated_logistic(x_raw, y, topsis_risk):
    folds = stratified_folds(y, PARAM["cv_folds"], SEED)
    pred = np.zeros(len(y))
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        xtr, xte, _, _ = safe_standardize(x_raw[train], x_raw[test])
        beta = fit_logistic_irls(xtr, y[train], PARAM["ridge"])
        pred[test] = predict_logistic(xte, beta)
    # 在0~1中搜索融合系数，使Brier分数最小。
    theta_grid = np.linspace(0, 1, 101)
    briers = [np.mean((theta * pred + (1 - theta) * topsis_risk - y) ** 2) for theta in theta_grid]
    theta = float(theta_grid[int(np.argmin(briers))])
    fused = theta * pred + (1 - theta) * topsis_risk
    return pred, fused, theta, classification_metrics(y, fused)


def matrix_power_psd(c: np.ndarray, power: float):
    vals, vecs = np.linalg.eigh(c)
    vals = np.clip(vals, 1e-8, None)
    return vecs @ np.diag(vals ** power) @ vecs.T


def coral_transform(xs: np.ndarray, xt: np.ndarray):
    """将源域协方差对齐到目标域；返回校正源域和对齐误差。"""
    ms, mt = np.mean(xs, axis=0), np.mean(xt, axis=0)
    xsc, xtc = xs - ms, xt - mt
    cs = np.cov(xsc, rowvar=False) + PARAM["coral_eps"] * np.eye(xs.shape[1])
    ct = np.cov(xtc, rowvar=False) + PARAM["coral_eps"] * np.eye(xt.shape[1])
    before = np.linalg.norm(cs - ct, ord="fro")
    aligned = xsc @ matrix_power_psd(cs, -0.5) @ matrix_power_psd(ct, 0.5) + mt
    after = np.linalg.norm(np.cov(aligned, rowvar=False) - ct, ord="fro")
    return aligned, before, after


def interpolate_loss(rate_table: pd.DataFrame, grade: str, rates: np.ndarray):
    """仅用NumPy线性插值；输入位于附件范围内，不做危险外推。"""
    col = grade if grade in {"A", "B", "C"} else "C"
    return np.interp(rates, rate_table["rate"].to_numpy(), rate_table[col].to_numpy())


def grade_thresholds(source_risk: np.ndarray, source_grades: pd.Series):
    """按附件1历史等级数量比例建立风险阈值，保证尺度可迁移。"""
    counts = source_grades.value_counts()
    n = len(source_grades)
    # 风险从低到高依次A/B/C/D。
    q_a = counts.get("A", 0) / n
    q_b = (counts.get("A", 0) + counts.get("B", 0)) / n
    q_c = (counts.get("A", 0) + counts.get("B", 0) + counts.get("C", 0)) / n
    return tuple(np.quantile(source_risk, [q_a, q_b, q_c]))


def assign_grade(risk: np.ndarray, cuts):
    c1, c2, c3 = cuts
    return np.where(risk <= c1, "A", np.where(risk <= c2, "B", np.where(risk <= c3, "C", "D")))


def best_rate_and_utility(risk, grades, rate_table, risk_aversion=None):
    """逐企业枚举附件3利率档，选择每万元风险调整收益最大的利率。"""
    lam = PARAM["risk_aversion"] if risk_aversion is None else risk_aversion
    rates = rate_table["rate"].to_numpy(dtype=float)
    best_rate = np.zeros(len(risk))
    best_loss = np.zeros(len(risk))
    best_net = np.full(len(risk), -np.inf)
    best_utility = np.full(len(risk), -np.inf)
    for i in range(len(risk)):
        losses = interpolate_loss(rate_table, str(grades[i]), rates)
        accept = 1.0 - losses
        net = accept * (rates * (1 - risk[i]) - risk[i] * PARAM["lgd"])
        expected_loss = accept * risk[i] * PARAM["lgd"]
        utility = net - lam * expected_loss
        k = int(np.argmax(utility))
        best_rate[i], best_loss[i] = rates[k], losses[k]
        best_net[i], best_utility[i] = net[k], utility[k]
    return best_rate, best_loss, best_net, best_utility


def allocate_loans(risk, grades, uncertainty, rate_table, budget, risk_aversion=None, force_full=False):
    """
    离散额度优化：
    1) 只考虑A/B/C且单位风险调整收益为正的企业；
    2) 先为最优企业配置最低10万元；
    3) 再按边际效用逐1万元增加，直至预算耗尽或达到额度上限。
    """
    rates, churn, net_per_wan, utility = best_rate_and_utility(risk, grades, rate_table, risk_aversion)
    uncertainty_norm = uncertainty / max(np.max(uncertainty), 1e-12)
    adjusted = utility - PARAM["uncertainty_eta"] * uncertainty_norm * np.maximum(np.abs(utility), 0.01)
    # 问题2、3题意给出“总额为1亿元”，采用强等式约束：即使部分边际收益较低，
    # 也需在A/B/C企业中选取相对最优者完成配置；问题1仍允许预算不完全使用。
    eligible = (grades != "D") & ((adjusted > 0) | force_full)
    # 风险越高、预测越不确定，允许的最大额度越低。
    caps = np.floor(10 + 90 * (1 - risk) ** 1.4 * (1 - 0.45 * uncertainty_norm)).astype(int)
    caps = np.clip(caps, PARAM["min_loan_wan"], PARAM["max_loan_wan"])
    # 若风险折减后的容量仍不足以承接强制预算，则从效用最高企业开始把上限放宽至100万元。
    if force_full and np.sum(caps[eligible]) < budget:
        need = int(budget - np.sum(caps[eligible]))
        for i in np.argsort(-adjusted):
            if eligible[i] and need > 0:
                add_cap = min(PARAM["max_loan_wan"] - caps[i], need)
                caps[i] += add_cap
                need -= add_cap
    loan = np.zeros(len(risk), dtype=int)
    order = np.argsort(-adjusted)
    remaining = int(budget)
    for i in order:
        if eligible[i] and remaining >= PARAM["min_loan_wan"]:
            loan[i] = PARAM["min_loan_wan"]
            remaining -= PARAM["min_loan_wan"]
    # 每增加1万元都投向当前边际效用最高且未达上限的企业。
    for i in order:
        if not eligible[i] or remaining <= 0:
            continue
        add = min(caps[i] - loan[i], remaining)
        if add > 0:
            loan[i] += int(add)
            remaining -= int(add)
    expected_net = loan * net_per_wan
    expected_loss = loan * (1 - churn) * risk * PARAM["lgd"]
    return {
        "loan": loan, "rate": rates, "churn": churn, "net": expected_net,
        "expected_loss": expected_loss, "utility_per_wan": adjusted,
        "cap": caps, "unused_budget": remaining,
    }


def infer_industry(name: str):
    name = str(name)
    rules = [
        ("医疗健康", ["医", "药", "生物", "健康", "卫生"]),
        ("住宿餐饮及生活服务", ["餐饮", "酒店", "宾馆", "旅游", "娱乐", "家政"]),
        ("交通物流", ["物流", "运输", "货运", "仓储", "快递"]),
        ("建筑房地产", ["建筑", "工程", "装饰", "房地产", "建材"]),
        ("科技信息服务", ["科技", "信息", "软件", "网络", "电子", "技术"]),
        ("制造业", ["制造", "机械", "设备", "加工", "材料", "工业"]),
        ("批发零售", ["商贸", "销售", "批发", "零售", "超市", "供应链"]),
    ]
    for industry, words in rules:
        if any(w in name for w in words):
            return industry
    return "其他/无法识别"


SHOCK = {
    "制造业": (-0.05, -0.15, -0.30),
    "批发零售": (-0.08, -0.20, -0.40),
    "住宿餐饮及生活服务": (-0.15, -0.40, -0.65),
    "建筑房地产": (-0.05, -0.18, -0.35),
    "交通物流": (-0.08, -0.25, -0.45),
    "科技信息服务": (-0.02, -0.08, -0.18),
    "医疗健康": (0.02, 0.05, -0.05),
    "其他/无法识别": (-0.05, -0.15, -0.30),
}


def simulate_shocks(df: pd.DataFrame, base_risk: np.ndarray):
    """行业直接冲击 + 集中度代理的一阶供应链传播 + Monte Carlo。"""
    industries = df["行业"].to_numpy()
    concentration = np.clip(df["上下游集中度"].to_numpy(float), 0, 1)
    volatility = df["经营波动率"].to_numpy(float)
    vol_scaled = (volatility - np.min(volatility)) / max(np.ptp(volatility), 1e-12)
    abnormal = np.clip(df["综合作废率"].to_numpy(float) + df["综合负数率"].to_numpy(float), 0, 1)
    exposure = np.clip(0.45 * concentration + 0.35 * vol_scaled + 0.20 * abnormal, 0, 1)
    s_count, n = PARAM["mc_scenarios"], len(df)
    shocks = np.zeros((s_count, n))
    unique_ind = sorted(set(industries))
    for s in range(s_count):
        direct = np.zeros(n)
        for ind in unique_ind:
            light, medium, severe = SHOCK[ind]
            # 70%中度附近，15%轻度、15%重度；加入小幅随机扰动。
            mode = RNG.choice([0, 1, 2], p=[0.15, 0.70, 0.15])
            center = [light, medium, severe][mode]
            draw = np.clip(RNG.normal(center, max(abs(center) * 0.18, 0.015)), -0.80, 0.20)
            mask = industries == ind
            # 负收入冲击转成正风险冲击；受益行业可能得到负风险冲击。
            direct[mask] = -draw * (0.45 + 0.55 * exposure[mask])
        # 附件交易对象大多不是样本企业，采用“行业共同冲击×企业集中度”作为一阶传播代理。
        group_mean = pd.Series(direct).groupby(industries).transform("mean").to_numpy()
        propagated = PARAM["propagation_rho"] * concentration * group_mean
        shocks[s] = np.clip(direct + propagated, -0.20, 1.00)
    logits = np.log(np.clip(base_risk, 1e-5, 1 - 1e-5) / np.clip(1 - base_risk, 1e-5, 1))
    stress_prob = sigmoid(logits[None, :] + PARAM["shock_gamma"] * shocks)
    return shocks, stress_prob, exposure


def cvar(losses: np.ndarray, alpha: float):
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    return var, float(np.mean(tail))


def svg_bar(labels, values, title, xlabel, output, caption, color="#2F75B5", top_n=20):
    order = np.argsort(values)[-top_n:]
    labels = [str(labels[i]) for i in order]
    values = np.asarray(values)[order]
    w, h, left, right, top, bottom = 1200, 720, 230, 80, 95, 100
    ph = h - top - bottom
    vmax = max(float(np.max(values)), 1e-9)
    row = ph / max(len(values), 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Microsoft YaHei,SimHei,sans-serif;fill:#243447}.t{font-size:28px;font-weight:bold}.l{font-size:14px}.c{font-size:15px;fill:#5b6770}</style>',
             f'<text x="{w/2}" y="42" text-anchor="middle" class="t">{title}</text>']
    for j, (lab, val) in enumerate(zip(labels, values)):
        y = top + j * row + row * 0.15
        bw = (w - left - right) * float(val) / vmax
        parts += [f'<text x="{left-12}" y="{y+row*0.52}" text-anchor="end" class="l">{lab}</text>',
                  f'<rect x="{left}" y="{y}" width="{max(bw,1)}" height="{row*0.65}" fill="{color}"/>',
                  f'<text x="{left+bw+7}" y="{y+row*0.52}" class="l">{val:.2f}</text>']
    parts += [f'<line x1="{left}" y1="{h-bottom+5}" x2="{w-right}" y2="{h-bottom+5}" stroke="#697386"/>',
              f'<text x="{(left+w-right)/2}" y="{h-60}" text-anchor="middle" class="c">{xlabel}</text>',
              f'<text x="{w/2}" y="{h-22}" text-anchor="middle" class="c">{caption}</text>', '</svg>']
    Path(output).write_text("".join(parts), encoding="utf-8")


def svg_hist(values, title, xlabel, output, caption, bins=18, color="#70AD47"):
    values = np.asarray(values, dtype=float)
    counts, edges = np.histogram(values, bins=bins)
    w, h, left, right, top, bottom = 1100, 650, 100, 70, 90, 110
    pw, ph = w-left-right, h-top-bottom
    ymax = max(np.max(counts), 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Microsoft YaHei,SimHei,sans-serif;fill:#243447}.t{font-size:28px;font-weight:bold}.l{font-size:14px}.c{font-size:15px;fill:#5b6770}</style>',
             f'<text x="{w/2}" y="42" text-anchor="middle" class="t">{title}</text>']
    bw = pw / len(counts)
    for i, c in enumerate(counts):
        bh = ph * c / ymax
        parts.append(f'<rect x="{left+i*bw+1}" y="{top+ph-bh}" width="{max(bw-2,1)}" height="{bh}" fill="{color}"/>')
    for k in range(6):
        idx = min(int(k*(len(edges)-1)/5), len(edges)-1)
        x = left + k*pw/5
        parts.append(f'<text x="{x}" y="{h-bottom+28}" text-anchor="middle" class="l">{edges[idx]:.2f}</text>')
    parts += [f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#697386"/>',
              f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#697386"/>',
              f'<text x="{left+pw/2}" y="{h-62}" text-anchor="middle" class="c">{xlabel}</text>',
              f'<text x="{w/2}" y="{h-22}" text-anchor="middle" class="c">{caption}</text>', '</svg>']
    Path(output).write_text("".join(parts), encoding="utf-8")


def solve():
    validate_parameters()
    print("[1/8] 读取数据...")
    df, rate_table = load_data()
    source = df[df["样本类型"] == "有信贷记录"].copy().reset_index(drop=True)
    target = df[df["样本类型"] == "无信贷记录"].copy().reset_index(drop=True)
    xs_raw = source[RISK_FEATURES].to_numpy(float)
    xt_raw = target[RISK_FEATURES].to_numpy(float)
    y = source["是否违约"].eq("是").astype(int).to_numpy()

    print("[2/8] 问题1：CRITIC-TOPSIS与Logistic监督校准...")
    safety, critic_w, _ = critic_topsis(xs_raw, DIRECTION)
    topsis_risk = 1 - safety
    cv_logit, fused_cv, theta, metrics = cross_validated_logistic(xs_raw, y, topsis_risk)
    xs_z, mean_s, std_s = safe_standardize(xs_raw)
    beta_final = fit_logistic_irls(xs_z, y, PARAM["ridge"])
    logit_full = predict_logistic(xs_z, beta_final)
    risk1 = np.clip(theta * logit_full + (1 - theta) * topsis_risk, 0.001, 0.999)
    grades1 = source["信誉评级"].fillna("D").astype(str).to_numpy()
    alloc1 = allocate_loans(risk1, grades1, np.zeros(len(source)), rate_table, PARAM["question1_budget_wan"], force_full=True)
    q1 = source[["企业代号", "企业名称", "信誉评级", "是否违约"]].copy()
    q1["TOPSIS安全度"] = safety
    q1["Logistic违约概率"] = logit_full
    q1["融合违约风险"] = risk1
    q1["是否放贷"] = np.where(alloc1["loan"] > 0, "是", "否")
    q1["贷款额度_万元"] = alloc1["loan"]
    q1["贷款年利率"] = alloc1["rate"]
    q1["客户流失率"] = alloc1["churn"]
    q1["预期损失_万元"] = alloc1["expected_loss"]
    q1["预期净收益_万元"] = alloc1["net"]
    q1.sort_values("融合违约风险").to_csv(OUT / "问题1_风险与信贷策略.csv", index=False, encoding="utf-8-sig")

    print("[3/8] 问题2：CORAL域适应与Bootstrap不确定性...")
    all_raw = np.vstack([xs_raw, xt_raw])
    all_z, all_mean, all_std = safe_standardize(all_raw)
    xs_common, xt_common = all_z[:len(source)], all_z[len(source):]
    xs_aligned, coral_before, coral_after = coral_transform(xs_common, xt_common)
    boot_pred = np.zeros((PARAM["bootstrap_times"], len(target)))
    for b in range(PARAM["bootstrap_times"]):
        idx = RNG.integers(0, len(source), len(source))
        # 若极小概率抽样后只有一个类别，重抽以保证模型可识别。
        while len(np.unique(y[idx])) < 2:
            idx = RNG.integers(0, len(source), len(source))
        beta_b = fit_logistic_irls(xs_aligned[idx], y[idx], PARAM["ridge"])
        boot_pred[b] = predict_logistic(xt_common, beta_b)
    risk2_mean = np.mean(boot_pred, axis=0)
    risk2_std = np.std(boot_pred, axis=0, ddof=1)
    risk2 = np.clip(risk2_mean + PARAM["uncertainty_eta"] * risk2_std, 0.001, 0.999)
    cuts = grade_thresholds(risk1, source["信誉评级"])
    grades2 = assign_grade(risk2, cuts)
    alloc2 = allocate_loans(risk2, grades2, risk2_std, rate_table, PARAM["question2_budget_wan"], force_full=True)
    q2 = target[["企业代号", "企业名称"]].copy()
    q2["CORAL平均违约风险"] = risk2_mean
    q2["预测不确定性"] = risk2_std
    q2["修正违约风险"] = risk2
    q2["预测信誉评级"] = grades2
    q2["是否放贷"] = np.where(alloc2["loan"] > 0, "是", "否")
    q2["贷款额度_万元"] = alloc2["loan"]
    q2["贷款年利率"] = alloc2["rate"]
    q2["客户流失率"] = alloc2["churn"]
    q2["预期损失_万元"] = alloc2["expected_loss"]
    q2["预期净收益_万元"] = alloc2["net"]
    q2.sort_values("修正违约风险").to_csv(OUT / "问题2_风险迁移与1亿元策略.csv", index=False, encoding="utf-8-sig")

    print("[4/8] 问题3：行业冲击、传播代理与Monte Carlo-CVaR...")
    target["行业"] = target["企业名称"].map(infer_industry)
    shocks, stress_prob, exposure = simulate_shocks(target, risk2)
    mean_stress = np.mean(stress_prob, axis=0)
    p95_stress = np.quantile(stress_prob, 0.95, axis=0)
    # 以均值和95%分位风险的加权值进行稳健配置。
    robust_risk = np.clip(0.55 * mean_stress + 0.45 * p95_stress, 0.001, 0.999)
    grades3 = assign_grade(robust_risk, cuts)
    alloc3 = allocate_loans(robust_risk, grades3, risk2_std, rate_table, PARAM["question2_budget_wan"], risk_aversion=0.55, force_full=True)
    accept3 = 1 - alloc3["churn"]
    accept2 = 1 - alloc2["churn"]
    baseline_scenario_losses = np.sum(
        alloc2["loan"][None, :] * accept2[None, :] * stress_prob * PARAM["lgd"], axis=1
    )
    scenario_losses = np.sum(
        alloc3["loan"][None, :] * accept3[None, :] * stress_prob * PARAM["lgd"], axis=1
    )
    baseline_var95, baseline_cvar95 = cvar(baseline_scenario_losses, PARAM["cvar_alpha"])
    var95, cvar95 = cvar(scenario_losses, PARAM["cvar_alpha"])
    q3 = target[["企业代号", "企业名称", "行业"]].copy()
    q3["问题2额度_万元"] = alloc2["loan"]
    q3["平均冲击强度"] = np.mean(shocks, axis=0)
    q3["企业暴露度"] = exposure
    q3["冲击前风险"] = risk2
    q3["冲击后平均风险"] = mean_stress
    q3["冲击后P95风险"] = p95_stress
    q3["稳健风险"] = robust_risk
    q3["调整后评级"] = grades3
    q3["调整后额度_万元"] = alloc3["loan"]
    q3["额度变化_万元"] = alloc3["loan"] - alloc2["loan"]
    q3["调整后利率"] = alloc3["rate"]
    q3["调整类别"] = np.where(q3["额度变化_万元"] > 0, "调增", np.where(q3["额度变化_万元"] < 0, "调减", "维持"))
    q3.loc[(q3["问题2额度_万元"] > 0) & (q3["调整后额度_万元"] == 0), "调整类别"] = "退出"
    q3.to_csv(OUT / "问题3_突发事件稳健调整策略.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"情景编号": np.arange(1, len(scenario_losses)+1), "问题2原方案损失_万元": baseline_scenario_losses, "问题3调整方案损失_万元": scenario_losses}).to_csv(
        OUT / "问题3_MonteCarlo情景损失.csv", index=False, encoding="utf-8-sig"
    )

    print("[5/8] 汇总模型评价与结果...")
    summary = {
        "随机种子": SEED,
        "问题1": {
            "交叉验证指标": metrics,
            "融合系数theta": theta,
            "预算_万元": PARAM["question1_budget_wan"],
            "实际贷款总额_万元": int(np.sum(alloc1["loan"])),
            "获贷企业数": int(np.sum(alloc1["loan"] > 0)),
            "预期净收益_万元": float(np.sum(alloc1["net"])),
            "预期损失_万元": float(np.sum(alloc1["expected_loss"])),
            "未使用预算_万元": int(alloc1["unused_budget"]),
        },
        "问题2": {
            "CORAL对齐前协方差差异": float(coral_before),
            "CORAL对齐后协方差差异": float(coral_after),
            "CORAL改善率": float(1 - coral_after / max(coral_before, 1e-12)),
            "评级阈值": [float(v) for v in cuts],
            "预算_万元": PARAM["question2_budget_wan"],
            "实际贷款总额_万元": int(np.sum(alloc2["loan"])),
            "获贷企业数": int(np.sum(alloc2["loan"] > 0)),
            "预期净收益_万元": float(np.sum(alloc2["net"])),
            "预期损失_万元": float(np.sum(alloc2["expected_loss"])),
            "未使用预算_万元": int(alloc2["unused_budget"]),
        },
        "问题3": {
            "MonteCarlo情景数": PARAM["mc_scenarios"],
            "调整后贷款总额_万元": int(np.sum(alloc3["loan"])),
            "获贷企业数": int(np.sum(alloc3["loan"] > 0)),
            "VaR95_万元": var95,
            "CVaR95_万元": cvar95,
            "问题2原方案冲击后VaR95_万元": baseline_var95,
            "问题2原方案冲击后CVaR95_万元": baseline_cvar95,
            "CVaR降低率": float((baseline_cvar95-cvar95)/max(baseline_cvar95,1e-12)),
            "情景平均损失_万元": float(np.mean(scenario_losses)),
            "调增企业数": int(np.sum(q3["调整类别"] == "调增")),
            "调减企业数": int(np.sum(q3["调整类别"] == "调减")),
            "退出企业数": int(np.sum(q3["调整类别"] == "退出")),
        },
        "参数": PARAM,
        "CRITIC权重": {name: float(w) for name, w in zip(RISK_FEATURES, critic_w)},
    }
    (OUT / "模型求解汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# 2020 C题模型求解摘要

## 问题1

- 五折交叉验证 AUC：{metrics['AUC']:.4f}
- Brier分数：{metrics['Brier']:.4f}
- 违约召回率：{metrics['Recall']:.2%}
- TOPSIS—Logistic最优融合系数：{theta:.2f}
- 基准预算：{PARAM['question1_budget_wan']}万元
- 实际配置：{np.sum(alloc1['loan'])}万元，共{np.sum(alloc1['loan']>0)}家企业
- 预期净收益：{np.sum(alloc1['net']):.2f}万元
- 预期损失：{np.sum(alloc1['expected_loss']):.2f}万元

## 问题2

- CORAL协方差差异：{coral_before:.4f} → {coral_after:.4f}
- 分布对齐改善率：{1-coral_after/max(coral_before,1e-12):.2%}
- 实际配置：{np.sum(alloc2['loan'])}万元，共{np.sum(alloc2['loan']>0)}家企业
- 预期净收益：{np.sum(alloc2['net']):.2f}万元
- 预期损失：{np.sum(alloc2['expected_loss']):.2f}万元

## 问题3

- Monte Carlo情景数：{PARAM['mc_scenarios']}
- 调整后配置：{np.sum(alloc3['loan'])}万元
- 情景平均损失：{np.mean(scenario_losses):.2f}万元
- 95% VaR：{var95:.2f}万元
- 95% CVaR：{cvar95:.2f}万元
- 问题2原方案在同一冲击下CVaR：{baseline_cvar95:.2f}万元
- 调整后CVaR降低率：{(baseline_cvar95-cvar95)/max(baseline_cvar95,1e-12):.2%}
- 调增/调减/退出企业：{np.sum(q3['调整类别']=='调增')}/ {np.sum(q3['调整类别']=='调减')}/ {np.sum(q3['调整类别']=='退出')}
"""
    (OUT / "模型求解摘要.md").write_text(report, encoding="utf-8")

    print("[6/8] 生成论文可视化...")
    svg_hist(risk1, "问题1：123家企业融合违约风险分布", "融合违约风险", OUT / "图1_问题1风险分布.svg",
             f"结论：风险中位数为{np.median(risk1):.3f}，模型应优先识别右侧高风险企业。")
    svg_bar(q1["企业代号"].to_numpy(), alloc1["loan"], "问题1：贷款额度最高的企业", "贷款额度（万元）",
            OUT / "图2_问题1额度配置.svg", f"结论：基准情景共配置{np.sum(alloc1['loan'])}万元。")
    grade_counts = pd.Series(grades2).value_counts().reindex(["A","B","C","D"], fill_value=0)
    svg_bar(grade_counts.index.to_numpy(), grade_counts.to_numpy(), "问题2：302家企业预测信誉等级", "企业数量",
            OUT / "图3_问题2预测评级.svg", "结论：D级企业按题意不进入基准贷款方案。", top_n=4, color="#5B9BD5")
    svg_bar(q2["企业代号"].to_numpy(), alloc2["loan"], "问题2：1亿元贷款额度配置", "贷款额度（万元）",
            OUT / "图4_问题2额度配置.svg", f"结论：总配置{np.sum(alloc2['loan'])}万元，获贷企业{np.sum(alloc2['loan']>0)}家。")
    change_abs = np.abs(q3["额度变化_万元"].to_numpy(float))
    svg_bar(q3["企业代号"].to_numpy(), change_abs, "问题3：突发事件下额度调整幅度", "|额度变化|（万元）",
            OUT / "图5_问题3额度调整.svg", f"结论：调减{np.sum(q3['调整类别']=='调减')}家，退出{np.sum(q3['调整类别']=='退出')}家。", color="#ED7D31")
    svg_hist(scenario_losses, "问题3：调整后Monte Carlo组合损失分布", "组合损失（万元）",
             OUT / "图6_问题3情景损失.svg", f"结论：CVaR由{baseline_cvar95:.2f}降至{cvar95:.2f}万元，降低{(baseline_cvar95-cvar95)/max(baseline_cvar95,1e-12):.2%}。", color="#C55A11")

    print("[7/8] 写出复现信息...")
    pd.DataFrame({"特征": RISK_FEATURES, "CRITIC权重": critic_w}).sort_values("CRITIC权重", ascending=False).to_csv(
        OUT / "问题1_CRITIC权重.csv", index=False, encoding="utf-8-sig"
    )
    print("[8/8] 完成。输出目录：", OUT)
    print(report)


if __name__ == "__main__":
    try:
        solve()
    except Exception as exc:
        print("\n程序运行失败：", exc)
        print("请检查文件路径、工作簿名称以及参数范围。详细信息如下：")
        traceback.print_exc()
        raise
