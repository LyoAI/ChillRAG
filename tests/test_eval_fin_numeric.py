"""FinNumeric: value parsing out of free text, unit conversion, and matching
at the coarser of the two claimed precisions, with the tolerance fallbacks."""

import pytest

from chillrag.eval import FinNumeric


def _score(answer: str, label: str, unit: str | None = None):
    row = {"index": 0, "question": "q", "answer": answer, "label": label}
    if unit:
        row["unit"] = unit
    return FinNumeric().score_row(row)


# ------------------------------------------------------------- parsing ----

def test_markdown_and_prose_do_not_hide_the_value():
    v = _score("**92,599,365.5 万元（约 9,259.94 亿元）**", "92,599,365.5万元")
    assert v.ok and v.detail == "exact"


def test_years_and_ids_do_not_capture_the_value():
    v = _score("天津银行(1578.HK)2024年末资产总额92,599,365.5万元", "92599365.5万元")
    assert v.ok, "a unit-bearing number outranks bare ones from the prose"


def test_accounting_parentheses_are_negative():
    assert _score("(1,234.5)", "-1234.5").ok


# -------------------------------------------------------------- matching --

def test_identical_values_match_exactly():
    assert _score("42", "42").detail == "exact"


def test_the_coarser_of_the_two_precisions_decides():
    # 3.51亿 claims ±0.005亿 = ±50万, so it agrees with a 35142万 reference.
    assert _score("3.51亿元", "35142万元").ok
    # And 11.0 writes zero claimed decimals, like any serialized integer.
    assert _score("11.0", "11").ok
    assert _score("11.44", "11.4").ok, "11.44 quantizes to the reference's 0.1 step"


def test_unit_conversion_within_the_money_dimension():
    v = _score("3.52亿", "35142万元")
    assert v.ok and v.detail == "rel-money", "0.16% off across units is inside the 1% bound"


def test_percent_matches_its_own_text_and_multiples_convert():
    assert _score("19.3%", "19.3%").ok
    v = _score("1.25倍", "125%")
    assert v.ok and v.detail == "exact", "1.25倍 converts to 125% in the rate dimension"


def test_the_unit_column_gives_a_bare_label_its_dimension():
    assert _score("19.3%", "19.3", unit="%").ok


def test_a_unit_written_in_the_label_beats_the_column():
    v = _score("902,561万元人民币", "90.3亿元", unit="万元")
    assert v.ok, "the label says 亿元; the column never rewrites it — 90.2561亿 fits ±0.1亿"


def test_a_rounding_fallback_covers_the_last_digit():
    assert _score("35142.1万元", "35142.05万元").detail == "round-tol"


def test_incompatible_dimensions_and_foreign_currency_fail():
    assert _score("3.5万美元", "24.5万元").detail == "unit-mismatch"
    v = _score("9259.94亿元", "92599365.5")
    assert not v.ok, "a bare reference implies its own unit; no conversion is guessed"


# ------------------------------------------------------------ abstention --

def test_an_na_reference_requires_abstention():
    assert _score("无", "N/A").detail == "abstain-correct"
    assert _score("123亿", "N/A").detail == "hallucinated"


def test_an_abstaining_answer_against_a_real_reference_fails():
    v = _score("未找到", "35142万元")
    assert not v.ok and v.detail == "no-value"


def test_a_non_numeric_answer_is_no_value_not_a_crash():
    assert _score("喜欢香蕉", "42").detail == "no-value"


def test_a_non_numeric_label_is_reported_not_matched():
    assert _score("42", "资产总额").detail == "label-not-numeric"


# ----------------------------------------------------------- aggregates --

def test_aggregate_adds_the_abstention_view():
    ev = FinNumeric()
    verdicts = [ev.score_row({"answer": "42", "label": "42"}),
                ev.score_row({"answer": "无", "label": "N/A"}),
                ev.score_row({"answer": "43", "label": "42"})]
    stats = ev.aggregate(verdicts)
    assert stats["reference-na"] == 1
    assert stats["abstained"] == 0 and stats["hallucinated"] == 0
    assert stats["accuracy"] == pytest.approx(2 / 3)
    assert stats["accuracy-excl-na"] == pytest.approx(1 / 2)
