"""Issue 4: the benchmark runs end to end on a small slice of both scenarios,
is deterministic, and the outcomes do not drift from what the 10k run in
bench/RESULTS.md measured."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal

import duckdb
import pandas as pd
import pytest

from bench import corrupt, score

# Expected outcome per defect type, the same in both scenarios: the share of
# each type that ends up restored, quarantined, neutralized or missed. These
# are the 10k numbers; a change in either direction should be looked at and
# RESULTS.md regenerated.
EXPECTED = {
    "type_pollution": {"quarantined": 1.0},
    "null": {"quarantined": 0.5, "neutralized": 0.5},
    "duplicate": {"restored": 1.0},
    "out_of_range": {"quarantined": 1.0},
    "unit_error": {"restored": 1.0},
    "future_date": {"neutralized": 1.0},
    "pre_1970_date": {"neutralized": 1.0},
    # Not in the contract: nothing repairs them. A x10 slip is still flagged
    # by the outlier rule; the other two look exactly like good data.
    "x10_error": {"missed": 1.0},
    "in_range_wrong": {"missed": 1.0},
    "day_month_swap": {"missed": 1.0},
}
UNDESCRIBED_RECALL = {"x10_error": 1.0, "in_range_wrong": 0.0, "day_month_swap": 0.0}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return score.score_all(rows=500, seed=11, rate=0.02, out=tmp_path_factory.mktemp("bench"))


@pytest.mark.parametrize("name", list(corrupt.SCENARIOS))
def test_every_defect_has_the_measured_outcome(results, name):
    r = results[name]
    sc = corrupt.SCENARIOS[name]
    assert r["defects"] == 10 * len(sc.defect_types)
    for kind in sc.defect_types:
        v = r["per_type"][kind]
        assert v["defects"] == 10, kind
        shares = {o: v[o] / 10 for o in score.OUTCOMES}
        want = {o: EXPECTED[kind].get(o, 0.0) for o in score.OUTCOMES}
        assert shares == want, kind
        if v["described"]:
            assert v["recall"] == 1.0 and v["precision"] == 1.0, kind
        else:
            assert v["precision"] is None
            assert v["recall"] == UNDESCRIBED_RECALL[kind], kind
    assert r["failing_checks_after"] == []
    assert r["duplicate_originals_kept"] == 10


@pytest.mark.parametrize("name", list(corrupt.SCENARIOS))
def test_quarantined_and_neutralized_are_not_counted_as_restored(results, name):
    r = results[name]
    t = r["totals"]
    assert r["restored_rate"] == t["restored"] / r["defects"]
    assert t["restored"] + t["quarantined"] + t["neutralized"] + t["wrong"] + t["missed"] == r["defects"]
    assert "fix_accuracy" not in r
    for v in r["per_type"].values():
        assert v["restored_rate"] == v["restored"] / v["defects"]
    text = score.render(results)
    assert "Restored rate" in text and "Fix accuracy" not in text


@pytest.mark.parametrize("name", list(corrupt.SCENARIOS))
def test_clean_cells_are_never_changed(results, name):
    r = results[name]
    assert r["false_changes"] == 0 and r["clean_rows_lost"] == 0
    assert r["collateral_changes"] == 0
    assert r["clean_outlier_findings"] == 0
    clean = r["clean_only"]
    assert clean["failing_checks"] == [] and clean["outlier_findings"] == 0
    assert clean["value_proposals"] == 0 and clean["cells_changed"] == 0
    assert clean["staged_cells_differing"] == 0


def test_typed_staging_keeps_source_types(tmp_path):
    """The typed table's DECIMAL, DATE, TIMESTAMPTZ and BIGINT columns come out
    of staging with the same types and exact values, including a rescaled
    amount (291920.00 / 1000 is 291.92, not a float and not text)."""
    score.score(rows=200, seed=5, rate=0.02, out=tmp_path, scenario="typed")
    con = duckdb.connect(str(tmp_path / "dirty_typed.duckdb"), read_only=True)
    try:
        types = {r[0]: r[1] for r in con.execute('DESCRIBE "sales_typed__shdi_staging"').fetchall()}
        rows = con.execute('SELECT id, day, amount, epoch_ms(ts) FROM "sales_typed__shdi_staging" ORDER BY id').fetchall()
    finally:
        con.close()
    assert types["id"] == "BIGINT"
    assert types["day"] == "DATE"
    assert types["amount"] == "DECIMAL(10,2)"
    assert types["ts"] == "TIMESTAMP WITH TIME ZONE"
    clean = corrupt.make_clean(200, 5, corrupt.TYPED)
    by_id = {int(r["id"]): r for r in clean.to_dict("records")}
    for i, day, amount, ms in rows:
        # Out-of-range dates are nulled (neutralized); every other cell is typed.
        assert isinstance(amount, Decimal) and (day is None or isinstance(day, dt.date))
        epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        assert ms == (by_id[i]["ts"] - epoch) // dt.timedelta(milliseconds=1)
    _, labels = corrupt.corrupt(clean, 5, 0.02, corrupt.TYPED)
    rescaled = {int(r["id"]) for r in labels.to_dict("records") if r["defect_type"] == "unit_error"}
    staged = {i: amount for i, _, amount, _ in rows}
    assert rescaled and all(staged[i] == by_id[i]["amount"] for i in rescaled)


def test_same_seed_same_scores(results, tmp_path):
    again = score.score_all(rows=500, seed=11, rate=0.02, out=tmp_path)
    strip = lambda r: {k: v for k, v in r.items() if k != "seconds"}  # noqa: E731
    for name in corrupt.SCENARIOS:
        assert strip(again[name]) == strip(results[name])


@pytest.mark.parametrize("name", list(corrupt.SCENARIOS))
def test_corruption_is_deterministic_and_labelled(name):
    sc = corrupt.SCENARIOS[name]
    clean = corrupt.make_clean(300, 3, sc)
    a_dirty, a_labels = corrupt.corrupt(clean, 3, 0.02, sc)
    b_dirty, b_labels = corrupt.corrupt(corrupt.make_clean(300, 3, sc), 3, 0.02, sc)
    pd.testing.assert_frame_equal(a_dirty, b_dirty)
    pd.testing.assert_frame_equal(a_labels, b_labels)
    # One defect per row, and every label points at the injected value.
    assert a_labels.source_row.is_unique
    for lab in a_labels.to_dict("records"):
        if lab["defect_type"] == "duplicate":
            src = int(lab["original_value"].split()[-1])
            assert a_dirty.loc[lab["source_row"]].tolist() == a_dirty.loc[src].tolist()
        elif lab["defect_type"] == "null":
            assert a_dirty.at[lab["source_row"], lab["column"]] is None
        else:
            assert a_dirty.at[lab["source_row"], lab["column"]] == lab["injected_value"]
    # Clean sales stay inside +-3 sigma, so the outlier rule has nothing to find.
    s = clean[sc.measure].map(lambda v: Decimal(str(v)))
    assert s.between(200, 800).all()


@pytest.mark.parametrize("name", list(corrupt.SCENARIOS))
def test_undescribed_defects_stay_inside_the_contract(name):
    """The undescribed defects only test what the pipeline cannot see if each
    one passes every rule the contract states."""
    sc = corrupt.SCENARIOS[name]
    clean = corrupt.make_clean(1000, 4, sc)
    _, labels = corrupt.corrupt(clean, 4, 0.02, sc)
    seen = set()
    for lab in labels.to_dict("records"):
        v, o = lab["injected_value"], lab["original_value"]
        if lab["defect_type"] in ("x10_error", "in_range_wrong"):
            assert 0 <= Decimal(str(v)) <= 10000 and Decimal(str(v)) != Decimal(str(o))
            seen.add(lab["defect_type"])
        elif lab["defect_type"] == "day_month_swap":
            d = v if isinstance(v, dt.date) else dt.date.fromisoformat(v)
            o = o if isinstance(o, dt.date) else dt.date.fromisoformat(o)
            assert d != o and (d.year, d.month, d.day) == (o.year, o.day, o.month)
            assert corrupt.START <= d <= corrupt.END
            seen.add("day_month_swap")
    assert seen == set(corrupt.UNDESCRIBED)


def test_number_words():
    assert corrupt.number_words(0) == "zero"
    assert corrupt.number_words(300) == "three hundred"
    assert corrupt.number_words(517) == "five hundred seventeen"
    assert corrupt.number_words(642) == "six hundred forty-two"
