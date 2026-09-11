"""第一问模型检验：独立回代、线性松弛下界、随机扰动及执行检验。

在项目目录执行：python problem1_validate.py
依赖：numpy、pandas、scipy、matplotlib、openpyxl。
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

# 兼容本次工作区内安装的求解依赖，普通环境直接使用已安装的依赖。
ROOT = Path(__file__).resolve().parent
LOCAL = ROOT.parent / ".validation-deps"
if LOCAL.is_dir():
    sys.path.insert(0, str(LOCAL))

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from problem1_solve import INPUT_XLSX, Parameters, read_input, solve_dispatch

OUT = ROOT / "results" / "第一问模型检验"
SEED = 20260911
REPEATS = 100
TOL = 1e-5


def audit(d, p):
    """从原始功率和调度结果独立回代，不使用求解摘要中的残差。"""
    g, c, z, w, e = (d[k].to_numpy(float) for k in
        ["购电量_kWh", "充电量_kWh", "放电量_kWh", "弃光量_kWh", "期末储电量_kWh"])
    load = d["小区负载"].to_numpy(float) * p.dt_hours
    pv = d["光伏发电预测功率"].to_numpy(float) * p.dt_hours
    rb = g + pv + z - load - c - w
    re = e - np.r_[p.soc_initial_kwh, e[:-1]] - p.eta_charge*c + z/p.eta_discharge
    violations = np.r_[-g, -c, -z, -w, w-pv, c-p.max_energy_kwh,
                       z-p.max_energy_kwh, p.soc_min_kwh-e, e-p.soc_max_kwh, 0]
    return {
        "平衡最大残差_千瓦时": float(np.abs(rb).max()),
        "平衡均方根残差_千瓦时": float(np.sqrt(np.mean(rb**2))),
        "储能递推最大残差_千瓦时": float(np.abs(re).max()),
        "终端偏差_千瓦时": float(abs(e[-1]-p.soc_initial_kwh)),
        "最大边界违反量_千瓦时": float(violations.max()),
        "同时充放电时段数": int(np.sum((c>TOL)&(z>TOL))),
        "全天能量恒等式残差_千瓦时": float(abs(g.sum()-(load-pv+w).sum()
            -(1-p.eta_charge)*c.sum()-(1/p.eta_discharge-1)*z.sum())),
    }


def lower_bound(d, p):
    """独立构造连续松弛：省去状态二元变量及互斥联动，所得最优值为下界。"""
    n = len(d)
    a = lil_matrix((2*n+1, 5*n))
    rhs = np.zeros(2*n+1)
    load = d["小区负载"].to_numpy(float)*p.dt_hours
    pv = d["光伏发电预测功率"].to_numpy(float)*p.dt_hours
    for t in range(n):
        a[t, t], a[t,n+t], a[t,2*n+t], a[t,3*n+t] = 1,-1,1,-1
        rhs[t] = load[t]-pv[t]
        a[n+t,4*n+t], a[n+t,n+t], a[n+t,2*n+t] = 1,-p.eta_charge,1/p.eta_discharge
        if t:
            a[n+t,4*n+t-1] = -1
        else:
            rhs[n+t] = p.soc_initial_kwh
    a[2*n,5*n-1] = 1
    rhs[2*n] = p.soc_initial_kwh
    bounds = ([(0,None)]*n + [(0,p.max_energy_kwh)]*(2*n)
              + [(0,float(v)) for v in pv] + [(p.soc_min_kwh,p.soc_max_kwh)]*n)
    result = linprog(np.r_[d["电价"].to_numpy(float),np.zeros(4*n)],
                     A_eq=a.tocsr(), b_eq=rhs, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError("独立松弛下界求解失败："+result.message)
    return float(result.fun)


def perturb(d, load_factor, pv_factor, price_factor=1):
    q = d.copy()
    q["小区负载"] *= load_factor
    q["光伏发电预测功率"] *= pv_factor
    q["电价"] *= price_factor
    q["负荷电量_kWh"] = q["小区负载"]/6
    q["光伏电量_kWh"] = q["光伏发电预测功率"]/6
    q["净负荷_kWh"] = q["负荷电量_kWh"]-q["光伏电量_kWh"]
    return q


def evaluate(q, base, p):
    """分别检验重优化、全计划冻结、仅固定电池调度三种口径。"""
    sol = solve_dispatch(q,p)
    checks = audit(sol.schedule,p)
    if any(v > TOL for v in checks.values()):
        raise AssertionError("扰动后的独立回代未通过")
    s, b = sol.schedule, base.schedule
    cost = sol.summary["optimized_grid_cost_yuan"]
    c0 = base.summary["optimized_grid_cost_yuan"]
    c = b["充电量_kWh"].to_numpy()
    z = b["放电量_kWh"].to_numpy()
    load = q["负荷电量_kWh"].to_numpy()
    pv = q["光伏电量_kWh"].to_numpy()
    need = load-pv+c-z
    grid, waste = np.maximum(need,0),np.maximum(-need,0)
    feasible = bool(np.max(waste-pv)<=TOL)
    execution_cost = float(grid @ q["电价"].to_numpy()) if feasible else np.nan
    # 全部冻结时，供给不足为负平衡残差；富余也单独计量。
    balance = b["购电量_kWh"].to_numpy()+pv+z-load-c-b["弃光量_kWh"].to_numpy()
    row = {
        "重优化成本_元": cost,
        "成本变化率_百分比": 100*(cost-c0)/c0,
        "同情景无储能成本_元": sol.summary["no_storage_grid_cost_yuan"],
        "同情景节费率_百分比": sol.summary["cost_saving_percent"],
        "购电总量变化率_百分比": 100*(sol.summary["optimized_grid_energy_kWh"]/
            base.summary["optimized_grid_energy_kWh"]-1),
        "电池调度相对变化_百分比": float(100*(np.abs(s["充电量_kWh"]-c).sum()
            +np.abs(s["放电量_kWh"]-z).sum())/(c.sum()+z.sum())),
        "全冻结缺电量_千瓦时": float(np.maximum(-balance,0).sum()),
        "全冻结富余量_千瓦时": float(np.maximum(balance,0).sum()),
        "固定电池可行": feasible,
        "固定电池补救成本_元": execution_cost,
        "固定电池后悔率_百分比": 100*(execution_cost-cost)/cost,
        "固定电池不可吸收电量_千瓦时": float(np.maximum(waste-pv,0).sum()),
        "独立回代最大违反量_千瓦时": max(checks.values()),
    }
    return row


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d, p = read_input(INPUT_XLSX), Parameters()
    base = solve_dispatch(d,p)
    checks = audit(base.schedule,p)
    assert all(v<=TOL for v in checks.values()), checks
    bound = lower_bound(d,p)
    cost = base.summary["optimized_grid_cost_yuan"]
    assert -TOL <= cost-bound <= 1e-4
    old = pd.read_csv(ROOT/"results"/"问题1_逐时段调度.csv")
    old_checks = audit(old,p)
    assert all(v<=TOL for v in old_checks.values()), old_checks
    old_cost = float((old["购电量_kWh"]*d["电价"]).sum())
    assert abs(old_cost-cost)<=1e-4
    summary = {
        "基准成本_元": cost, "无储能成本_元":base.summary["no_storage_grid_cost_yuan"],
        "节费率_百分比":base.summary["cost_saving_percent"],
        "独立线性松弛下界_元":bound, "下界绝对差_元":cost-bound,
        "下界相对差_百分比":100*(cost-bound)/cost,
        "原结果成本复现差_元":abs(old_cost-cost), "检验容差_千瓦时":TOL,
        "基准独立回代":checks, "原保存结果独立回代":old_checks,
        "随机种子":SEED,"每组重复次数":REPEATS,
    }
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    rng = np.random.default_rng(SEED)
    rows=[]
    # 有界均匀相对扰动：幅度为最大绝对比例，标准差为幅度除以根号三。
    # 小时块保持六个时段同向；两个变量的扰动彼此独立；夜间零光伏保持为零。
    for kind in ["逐时段独立","一小时相关块"]:
        count=144 if kind=="逐时段独立" else 24
        noise=rng.uniform(-1,1,(REPEATS,2,count))
        if count==24:
            noise=np.repeat(noise,6,axis=2)
        for level in [.05,.10,.20]:
            for i in range(REPEATS):
                q=perturb(d,1+level*noise[i,0],1+level*noise[i,1])
                row=evaluate(q,base,p)
                rows.append({"扰动类型":kind,"扰动幅度_百分比":100*level,"样本序号":i+1,**row})
            print(f"已完成：{kind}，幅度 {level:.0%}，{REPEATS} 个情景。",flush=True)
    scenarios=pd.DataFrame(rows)
    scenarios.to_csv(OUT/"随机扰动逐情景结果.csv",index=False,encoding="utf-8-sig")
    grouped=[]
    for (kind,level), group in scenarios.groupby(["扰动类型","扰动幅度_百分比"],sort=False):
        change=group["成本变化率_百分比"]
        regret=group.loc[group["固定电池可行"],"固定电池后悔率_百分比"]
        grouped.append({"扰动类型":kind,"扰动幅度_百分比":level,
            "情景数":len(group),"成本变化均值_百分比":change.mean(),
            "成本变化标准差_百分点":change.std(ddof=1),
            "成本变化百分位2.5":change.quantile(.025),"成本变化百分位97.5":change.quantile(.975),
            "最大绝对成本变化_百分比":change.abs().max(),
            "平均节费率_百分比":group["同情景节费率_百分比"].mean(),
            "最小节费率_百分比":group["同情景节费率_百分比"].min(),
            "平均电池调度变化_百分比":group["电池调度相对变化_百分比"].mean(),
            "全冻结平均缺电量_千瓦时":group["全冻结缺电量_千瓦时"].mean(),
            "固定电池可行率_百分比":100*group["固定电池可行"].mean(),
            "可行情景平均后悔率_百分比":regret.mean(),
            "可行情景最大后悔率_百分比":regret.max(),
            "最大回代违反量_千瓦时":group["独立回代最大违反量_千瓦时"].max()})
    agg=pd.DataFrame(grouped)
    agg.to_csv(OUT/"随机扰动汇总.csv",index=False,encoding="utf-8-sig")
    stresses=[]
    for name,lf,pf,price in [("负荷增加百分之十",1.1,1,1),("光伏减少百分之十",1,.9,1),
            ("负荷增加且光伏减少百分之十",1.1,.9,1),("负荷减少且光伏增加百分之十",.9,1.1,1),
            ("电价整体增加百分之十",1,1,1.1),("电价整体减少百分之十",1,1,.9)]:
        stresses.append({"情景":name,**evaluate(perturb(d,lf,pf,price),base,p)})
    stress=pd.DataFrame(stresses)
    stress.to_csv(OUT/"系统性偏差压力检验.csv",index=False,encoding="utf-8-sig")
    sensitivity=[]
    for eta in [.8,.85,.9,np.sqrt(.9),.95,1.]:
        param=replace(p,eta_charge=eta,eta_discharge=eta)
        solution=solve_dispatch(d,param)
        assert all(v<=TOL for v in audit(solution.schedule,param).values())
        value=solution.summary["optimized_grid_cost_yuan"]
        sensitivity.append({"单程效率":eta,"往返效率":eta**2,"成本_元":value,
                            "相对主方案成本变化_百分比":100*(value/cost-1)})
    sens=pd.DataFrame(sensitivity)
    sens.to_csv(OUT/"效率假设检验.csv",index=False,encoding="utf-8-sig")
    summary["输入文件校验值"]=hashlib.sha256(INPUT_XLSX.read_bytes()).hexdigest()
    summary["运行环境"]={"python":platform.python_version(),"numpy":np.__version__,
                        "pandas":pd.__version__,"scipy":scipy.__version__}
    (OUT/"有效性检验摘要.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(agg.to_string(index=False),flush=True)
    print(stress[["情景","成本变化率_百分比","固定电池可行"]].to_string(index=False),flush=True)
    print(sens.to_string(index=False),flush=True)
    print("全部数值检验运行完成。",flush=True)


if __name__=="__main__":
    main()
