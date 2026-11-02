"""Connects the modules: scan file -> text -> question-wise marks."""
from modules import evaluator, ocr, parser, preprocessing


# --------------------------------------------------------------------------
# Heavy models are created once and kept in app.extensions.
# Tests can put fake objects into app.extensions to avoid loading models.
# --------------------------------------------------------------------------
def get_ocr_engine(app):
    if app.extensions.get("ocr_engine") is None:
        app.extensions["ocr_engine"] = ocr.create_engine(
            app.config["OCR_ENGINE"], app.config["TROCR_MODEL"])
    return app.extensions["ocr_engine"]


def get_similarity(app):
    if app.extensions.get("similarity") is None:
        app.extensions["similarity"] = evaluator.create_similarity(
            app.config["SIMILARITY_BACKEND"], app.config["SBERT_MODEL"])
    return app.extensions["similarity"]


def extract_text(path, engine, dpi=200):
    """Preprocess every page, read every text line, return the full text."""
    all_lines = []
    for page in preprocessing.load_pages(path, dpi=dpi):
        clean, binary, _angle = preprocessing.preprocess_page(page)
        line_images = preprocessing.segment_lines(clean, binary)
        if line_images:
            all_lines.extend(engine.read_lines(line_images))
    text = "\n".join(t for t in all_lines if t and t.strip())
    if not text.strip():
        raise ValueError(
            "No handwriting could be read from this file. Check that the scan is clear, "
            "upright and not blank.")
    return text


def evaluate_sheet_text(text, questions, similarity, cfg):
    """Return one result dict per question in the marking scheme."""
    answers = parser.split_answers(text)
    results = []
    for q in questions:
        answer = answers.get(q.number, "")
        res = evaluator.evaluate_answer(
            answer, q.model_answer, q.keyword_list, q.max_marks, similarity, cfg)
        res.update({"question_id": q.id, "question_no": q.number, "extracted_text": answer})
        results.append(res)
    return results
