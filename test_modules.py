"""Unit tests for the processing modules (no web server, no neural models)."""
import numpy as np
import pytest

from config import Config
from modules import evaluator, parser, preprocessing, scheme
from tests.conftest import make_sheet_image

CFG = {k: getattr(Config, k) for k in ("SIM_LOW", "SIM_HIGH", "SEMANTIC_WEIGHT", "KEYWORD_WEIGHT", "MARK_STEP")}


# ---------------------------------------------------------------- parser
def test_split_answers_basic():
    text = "Name: Ravi\nQ1. first answer\nmore text\nQ2) second answer\nAns 3: third"
    assert parser.split_answers(text) == {1: "first answer more text", 2: "second answer", 3: "third"}


def test_split_answers_ocr_digit_confusion():
    assert 1 in parser.split_answers("QI. answer one")
    assert 1 in parser.split_answers("Ql) answer one")


def test_split_answers_ignores_text_before_first_label_and_handles_empty():
    assert parser.split_answers("just a header line") == {}
    assert parser.split_answers("") == {}


# ---------------------------------------------------------------- scheme
def test_scheme_valid():
    rows = scheme.parse_scheme_csv("question_no,max_marks,model_answer,keywords\n1,5,ans,a;b\n")
    assert rows == [{"number": 1, "max_marks": 5.0, "model_answer": "ans", "keywords": "a;b"}]


@pytest.mark.parametrize("bad", [
    "",
    "question_no,max_marks\n1,5\n",
    "question_no,max_marks,model_answer,keywords\nx,5,ans,k\n",
    "question_no,max_marks,model_answer,keywords\n1,-2,ans,k\n",
    "question_no,max_marks,model_answer,keywords\n1,5,,k\n",
    "question_no,max_marks,model_answer,keywords\n1,5,a,k\n1,5,b,k\n",
    "question_no,max_marks,model_answer,keywords\n",
])
def test_scheme_invalid(bad):
    with pytest.raises(scheme.SchemeError):
        scheme.parse_scheme_csv(bad)


# ------------------------------------------------------------- evaluator
def test_keyword_match_with_plural_forms():
    frac, matched, missing = evaluator.keyword_match("Tables hold rows", ["table", "row", "null"])
    assert matched == ["table", "row"] and missing == ["null"] and frac == pytest.approx(2 / 3)


def test_stemmer_groups_word_forms():
    assert evaluator.stem("tables") == evaluator.stem("table")
    assert evaluator.stem("processes") == evaluator.stem("process")
    assert evaluator.stem("uniquely") == evaluator.stem("unique")
    assert evaluator.stem("stored") == evaluator.stem("storing") == evaluator.stem("store")
    assert evaluator.stem("plan") != evaluator.stem("plant")


def test_marks_zero_for_empty_answer_and_full_for_identical_answer():
    sim = evaluator.BagOfWordsSimilarity()
    model = "A primary key uniquely identifies each row in a table."
    empty = evaluator.evaluate_answer("", model, ["unique"], 5, sim, CFG)
    assert empty["ai_marks"] == 0
    same = evaluator.evaluate_answer(model, model, ["row", "table"], 5, sim, CFG)
    assert same["ai_marks"] == 5


def test_marks_never_exceed_max_and_use_half_steps():
    for s in (0.0, 0.2, 0.5, 0.65, 0.95, 1.0):
        for k in (0.0, 0.5, 1.0):
            m = evaluator.calculate_marks(s, k, True, 5, CFG)
            assert 0 <= m <= 5 and (m * 2) == int(m * 2)


def test_unrelated_answer_gets_low_marks():
    sim = evaluator.BagOfWordsSimilarity()
    res = evaluator.evaluate_answer("the weather is sunny today", "A primary key uniquely identifies each row.",
                                    ["unique", "row"], 5, sim, CFG)
    assert res["ai_marks"] <= 1


def test_no_keywords_uses_semantic_share_only():
    assert evaluator.calculate_marks(0.80, 0.0, False, 10, CFG) == 10


def test_bad_backend_name():
    with pytest.raises(ValueError):
        evaluator.create_similarity("nope")


# --------------------------------------------------------- preprocessing
def test_segment_lines_finds_each_text_line():
    clean, binary, _ = preprocessing.preprocess_page(make_sheet_image(n_lines=4))
    assert len(preprocessing.segment_lines(clean, binary)) == 4


def test_skewed_page_is_corrected_and_still_segments():
    img = make_sheet_image(n_lines=4, angle=3.0)
    clean, binary, angle = preprocessing.preprocess_page(img)
    assert abs(abs(angle) - 3.0) <= 1.0
    assert len(preprocessing.segment_lines(clean, binary)) == 4


def test_blank_page_gives_no_lines():
    blank = np.full((1000, 1600, 3), 255, dtype=np.uint8)
    clean, binary, _ = preprocessing.preprocess_page(blank)
    assert preprocessing.segment_lines(clean, binary) == []


def test_unreadable_image_file_raises(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        preprocessing.load_pages(bad)


def test_pdf_pages_are_loaded(tmp_path):
    fitz = pytest.importorskip("fitz")  # PyMuPDF

    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page()
        page.insert_text((72, 100), "Q1. hello world", fontsize=24)
    path = tmp_path / "two.pdf"
    doc.save(str(path))
    doc.close()
    assert len(preprocessing.load_pages(path)) == 2
