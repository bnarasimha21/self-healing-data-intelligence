"""Statistical anomaly findings on measure columns.

Only columns the contract marks `measure: true` are scanned, so keys such as
`id` are never reported as outliers. The rule is the modified z-score of
Iglewicz and Hoaglin: 0.6745 * |x - median| / MAD, flagged above
`anomaly.mad_threshold` (3.5 by default). Unlike a mean/std z-score it is not
masked by the outlier itself, so it still fires on a 10-row table, where a
plain |z| > 3 with ddof=0 can never exceed sqrt(n - 1) = 3.

Findings are review items. They never produce repair proposals; range repairs
come from the contract's min and max.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .checks import ParsedFrame, parse_frame, row_key
from .contract import Contract
from .values import INVALID, to_jsonable


def detect_anomalies(df: pd.DataFrame, contract: Contract, parsed: Optional[ParsedFrame] = None) -> List[Dict[str, Any]]:
    if parsed is None:
        parsed = parse_frame(df, contract)
    settings = contract.anomaly
    findings: List[Dict[str, Any]] = []
    for name in contract.measures():
        if name not in parsed:
            continue
        rows = [i for i, p in enumerate(parsed[name]) if p.value is not None and p.value is not INVALID]
        if len(rows) < settings.min_rows:
            continue
        vals = np.array([float(parsed[name][i].value) for i in rows])
        median = float(np.median(vals))
        mad = float(np.median(np.abs(vals - median)))
        if mad == 0.0:
            # More than half the values are identical. Any other value would get
            # an infinite score, which says nothing useful, so the column is skipped.
            continue
        scores = 0.6745 * np.abs(vals - median) / mad
        flagged = [(rows[k], float(scores[k])) for k in range(len(rows)) if scores[k] > settings.mad_threshold]
        if flagged:
            findings.append(
                to_jsonable(
                    {
                        "check_id": f"outlier_mad:{name}",
                        "column": name,
                        "rule": f"modified z-score > {settings.mad_threshold}",
                        "median": median,
                        "mad": mad,
                        "rows": [int(df.index[r]) for r, _ in flagged],
                        "row_keys": [row_key(df, r, contract) for r, _ in flagged],
                        "scores": [round(s, 2) for _, s in flagged],
                    }
                )
            )
    return findings
