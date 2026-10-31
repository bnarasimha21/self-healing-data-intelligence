"""Issue 2: the data contract replaces name-based heuristics."""
from __future__ import annotations

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from shdi.anomaly import detect_anomalies
from shdi.checks import run_checks
from shdi.connectors.csv import read_csv_text
from shdi.contract import Contract, load_contract
from shdi.repairs import propose_repairs

from conftest import AS_OF, CONTRACT, FIXTURES, ROOT

ORIGINAL_SAMPLE = """id,date,sales,region,notes
1,2024-01-05,100,North,ok
2,01/06/2024,200,North,ok
3,2024/07/01,three hundred,South,ok
4,2050-12-01,1000,East,ok
5,1960-01-01,50,West,ok
5,1960-01-01,50,West,ok
6,2025-01-01,400000,North,ok
7,,700,South,ok
8,2024-02-29,80,East,ok
9,2024-03-10,100,West,ok
"""


@pytest.fixture
def df():
    return read_csv_text(FIXTURES / "dirty_sales.csv")


@pytest.fixture
def contract():
    return load_contract(CONTRACT)


def failing_ids(results):
    return {r.check_id for r in results if not r.passed}


def test_fixture_matches_example_sample():
    assert (FIXTURES / "dirty_sales.csv").read_bytes() == (ROOT / "examples" / "dirty_sales.csv").read_bytes()
    # The example keeps the original 10 rows and adds two (missing region, negative sales).
    assert (ROOT / "examples" / "dirty_sales.csv").read_text().startswith(ORIGINAL_SAMPLE)


def test_sales_max_violation_reported_on_first_validation(df, contract):
    res = {r.check_id: r for r in run_checks(df, contract, AS_OF)}
    mx = res["max:sales"]
    assert not mx.passed and mx.severity == "error"
    assert {"_row": 6, "id": 6} in mx.failing_row_keys
    # The text-polluted value is a type failure on the same first pass, not a 0.
    assert res["type:sales"].failing_row_keys == [{"_row": 2, "id": 3}]


def test_primary_key_flags_duplicated_id_and_id_is_never_an_outlier(df, contract):
    res = {r.check_id: r for r in run_checks(df, contract, AS_OF)}
    pk = res["primary_key:id"]
    assert not pk.passed
    assert sorted(k["id"] for k in pk.failing_row_keys) == [5, 5]
    findings = detect_anomalies(df, contract)
    assert findings, "the 400000 sale should be an outlier"
    assert all(f["column"] == "sales" for f in findings)
    assert not any(f["column"] == "id" for f in findings)


def test_outlier_rule_fires_on_ten_rows_where_old_zscore_could_not():
    # Old rule: |z| > 3 with ddof=0 is capped at sqrt(n-1) = 3 for n=10, so it never fired.
    df = pd.read_csv(pd.io.common.StringIO(ORIGINAL_SAMPLE), dtype=str, keep_default_na=False, na_values=[""])
    contract = load_contract(CONTRACT)
    findings = detect_anomalies(df, contract)
    rows = {r for f in findings for r in f["rows"]}
    assert 6 in rows  # 400000
    assert all(f["column"] == "sales" for f in findings)


def test_no_outliers_on_clean_measures():
    contract = load_contract(CONTRACT)
    df = pd.DataFrame(
        {
            "id": [str(i) for i in range(1, 21)],
            "date": ["2024-01-01"] * 20,
            "sales": [str(100 + (i % 7) * 10) for i in range(20)],
            "region": ["North"] * 20,
            "notes": ["ok"] * 20,
        }
    )
    assert detect_anomalies(df, contract) == []
    assert failing_ids(run_checks(df, contract, AS_OF)) == set()


def test_null_date_is_left_alone_not_filled(df, contract):
    """The old rule 'filled' dates with NaT, a no-op. Now a null in a nullable
    date is valid and no proposal targets it."""
    checks = run_checks(df, contract, AS_OF)
    proposals = propose_repairs(df, contract, checks, AS_OF)
    row7 = [p for p in proposals if 7 in p.rows and p.column == "date"]
    assert row7 == []
    assert not any(p.rule == "fill_default" and p.column == "date" for p in proposals)


def test_contract_rejects_unknown_fields(tmp_path):
    bad = yaml.safe_load(CONTRACT.read_text())
    bad["columns"]["sales"]["maximum"] = 5
    with pytest.raises(ValidationError):
        Contract.model_validate(bad)
    bad = yaml.safe_load(CONTRACT.read_text())
    bad["columns"]["sales"]["checks"]["typo"] = {"severity": "warn"}
    with pytest.raises(ValidationError):
        Contract.model_validate(bad)


def test_on_violation_override_changes_the_proposal(df):
    data = yaml.safe_load(CONTRACT.read_text())
    data["columns"]["sales"]["checks"]["max"] = {"on_violation": "quarantine"}
    data["columns"]["region"]["checks"] = {"not_null": {"on_violation": "report_only"}}
    contract = Contract.model_validate(data)
    checks = run_checks(df, contract, AS_OF)
    proposals = propose_repairs(df, contract, checks, AS_OF)
    assert any(p.rule == "quarantine_row" and p.rows == [6] for p in proposals)
    assert not any(p.rule == "rescale" for p in proposals)
    assert not any(p.column == "region" for p in proposals)
