#!/usr/bin/env python3
"""用已保存逐时段结果刷新运行摘要中的独立校验字段。"""

import json
from pathlib import Path

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import q234_solve as solver


frame = pd.read_csv(ROOT / "results" / "问题2至4_逐时段完整结果.csv", parse_dates=["日期"])
checks = {name: solver.validate_schedule(group, solver.Parameters()) for name, group in frame.groupby("策略")}

summary_path = ROOT / "results" / "问题2至4_运行摘要.json"
payload = json.loads(summary_path.read_text(encoding="utf-8"))
payload["checks"].update(checks)
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
