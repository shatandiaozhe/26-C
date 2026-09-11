from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "附件"


def scalar(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value


def inspect_sheet(path: Path, sheet_name: str) -> dict:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    trimmed = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    sample = trimmed.head(8).replace({np.nan: None}).values.tolist()
    return {
        "工作表": sheet_name,
        "原始行列": list(raw.shape),
        "有效行列": list(trimmed.shape),
        "非空单元格": int(raw.notna().sum().sum()),
        "样例": [[scalar(v) for v in row] for row in sample],
    }


def main() -> None:
    reports = []
    for path in sorted(ROOT.glob("附件[1-4].xlsx")):
        excel = pd.ExcelFile(path)
        reports.append(
            {
                "文件": path.name,
                "工作表数量": len(excel.sheet_names),
                "工作表": [inspect_sheet(path, name) for name in excel.sheet_names],
            }
        )
    output = json.dumps(reports, ensure_ascii=False, indent=2)
    (ROOT.parent.parent / "附件结构审计.json").write_text(output, encoding="utf-8")
    print("附件结构审计已生成")


if __name__ == "__main__":
    main()
