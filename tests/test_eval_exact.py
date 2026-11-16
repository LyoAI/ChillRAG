"""Exact: case/punctuation/space are presentation; the content must agree."""

from chillrag.eval import Exact


def _score(answer: str, label: str):
    return Exact().score_row({"index": 0, "question": "q", "answer": answer, "label": label})


def test_choice_letters_match_across_case_and_form():
    assert _score("B.", "b").ok
    assert _score("A。", "A").ok


def test_noise_is_stripped_but_content_is_never_fuzzy():
    assert _score("招商银行", "招 商 银行").ok
    assert not _score("招商银行", "招商行者").ok


def test_an_abstaining_reference_requires_abstention():
    assert _score("N/A", "N/A").detail == "abstain-correct"
    assert _score("B", "N/A").detail == "hallucinated"


def test_an_empty_answer_against_a_real_label_is_no_answer():
    v = _score("", "B")
    assert not v.ok and v.detail == "no-answer"
