"""Issue 4: the benchmark runs end to end on a small slice, is deterministic,
and the scores do not regress below what the 10k run measured."""
from __future__ import annotations

import pandas as pd
import pytest

from bench import corrupt, score

# Floors from the 10k-row run in bench/RESULTS.md. Detection is exact on every
# type; dates outside the range are nulled (neutralized), so their fix accuracy is 0.
FIX_FLOORS = {
    "type_pollution": 1.0,
    "null": 0.5,
    "duplicate": 1.0,
    "out_of_range": 1.0,
    "unit_error": 1.0,
    "future_date": 0.0,
    "pre_1970_date": 0.0,
}


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return score.score(rows=500, seed=11, rate=0.02, out=tmp_path_factory.mktemp("bench"))


def test_every_defect_is_detected_and_handled(result):
    assert result["defects"] == 7 * 10
    for kind in corrupt.DEFECT_TYPES:
        v = result["per_type"][kind]
        assert v["defects"] == 10
        assert v["recall"] == 1.0 and v["precision"] == 1.0, kind
        assert v["fix_accuracy"] >= FIX_FLOORS[kind], kind
        assert v["wrong"] == 0 and v["missed"] == 0, kind
    assert result["failing_checks_after"] == []
    assert result["duplicate_originals_kept"] == 10


def test_clean_cells_are_never_changed(result):
    assert result["false_changes"] == 0 and result["clean_rows_lost"] == 0
    assert result["clean_outlier_findings"] == 0
    clean = result["clean_only"]
    assert clean["failing_checks"] == [] and clean["outlier_findings"] == 0
    assert clean["value_proposals"] == 0 and clean["cells_changed"] == 0


def test_same_seed_same_scores(result, tmp_path):
    again = score.score(rows=500, seed=11, rate=0.02, out=tmp_path)
    strip = lambda r: {k: v for k, v in r.items() if k != "seconds"}  # noqa: E731
    assert strip(again) == strip(result)


def test_corruption_is_deterministic_and_labelled():
    clean = corrupt.make_clean(300, 3)
    a_dirty, a_labels = corrupt.corrupt(clean, 3, 0.02)
    b_dirty, b_labels = corrupt.corrupt(corrupt.make_clean(300, 3), 3, 0.02)
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
    s = clean.sales.astype(int)
    assert s.between(200, 800).all()


def test_number_words():
    assert corrupt.number_words(0) == "zero"
    assert corrupt.number_words(300) == "three hundred"
    assert corrupt.number_words(517) == "five hundred seventeen"
    assert corrupt.number_words(642) == "six hundred forty-two"
