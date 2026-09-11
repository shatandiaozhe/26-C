#!/usr/bin/env python3
"""同步问题二至四重构后的 AI 自查表结论。"""

from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "2026数学建模国赛AI自查表_已核查（第2至4项）.docx"
path = ROOT / "2026数学建模国赛AI自查表_已核查（第2至4项）_模型修订版.docx"
document = Document(source)
table = document.tables[0]

updates = {
    10: (
        "合格",
        "模型假设.md 第5条；q234_solve.py 的 _dispatch_milp、solve_day_ahead、solve_rolling；独立跨日审计",
        "问题2—4已统一为混合整数规划；二元变量与有限功率上界严格保证充放电互斥。SOC按实际末状态传递至次日，不再逐日强制首末相等，仅在整个计算区间末端回到6000 kWh。",
        "保留跨日衔接、全年末端状态和二元互斥的自动化回归测试。",
    ),
    13: (
        "合格（并网容量待说明）",
        "q234_solve.py 的 _dispatch_milp；问题2至4_求解说明.md 关键口径",
        "SOC、功率、效率、5倍、1.5倍、0.5倍、弃光、禁售和问题1首末相等均已纳入；问题2—4改为跨日连续且仅全年末端回归，并以二元状态严格互斥。题目未给并网购电功率上限。",
        "论文声明未设置并网购电功率上限的假设，并在获得设备或并网参数后补做上限敏感性分析。",
    ),
    15: (
        "合格",
        "problem1_solve.py、q234_solve.py；问题2至4_运行摘要.json",
        "问题1和问题2—4均采用HiGHS混合整数规划。问题2—4执行两层词典序优化：首层最小化题目费用，次层在首层最优费用容差内最小化吞吐量；结果另行独立回代。",
        "持续记录求解状态、最优容差、约束残差、跨日衔接和全年末端SOC。",
    ),
    16: (
        "存在问题（仅剩分位数选参）",
        "q234_solve.py 参数；docs/七_结果分析阶段.md 7.6",
        "原无标定依据的0.005元/kWh吞吐成本已从首层目标删除，改为无货币权重的第二层吞吐量最小化。当前仍需在混合整数版本上重新系统选择0.80/0.20等预测分位数。",
        "按时间顺序样本外总成本完成分位数选参后定稿重跑。",
    ),
}

for row_index, values in updates.items():
    row = table.rows[row_index]
    for column, text in zip(range(2, 6), values):
        row.cells[column].text = text

document.save(path)
