"""Central settings for the Handwritten Exam Evaluator.

Every value can be changed here or through an environment variable.
"""
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # --- Flask / security -------------------------------------------------
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-before-real-use")
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB per upload

    # --- Database ----------------------------------------------------------
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + os.path.join(BASE_DIR, "instance", "exam_evaluator.db"),
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Files -------------------------------------------------------------
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
    ALLOWED_SHEET_EXT = {"pdf", "png", "jpg", "jpeg"}

    # --- OCR ---------------------------------------------------------------
    # "trocr"     : handwriting model (needs torch + transformers)
    # "tesseract" : printed / very neat text (needs the Tesseract program)
    OCR_ENGINE = os.environ.get("OCR_ENGINE", "trocr")
    TROCR_MODEL = os.environ.get("TROCR_MODEL", "microsoft/trocr-base-handwritten")
    PDF_RENDER_DPI = 200

    # --- Answer comparison -------------------------------------------------
    # "sbert" : Sentence-BERT embeddings (needs sentence-transformers)
    # "bow"   : simple word-overlap similarity (no extra install, weaker)
    SIMILARITY_BACKEND = os.environ.get("SIMILARITY_BACKEND", "sbert")
    SBERT_MODEL = os.environ.get("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # --- Marking rule ------------------------------------------------------
    # Similarity at or below SIM_LOW earns 0 of the semantic share.
    # Similarity at or above SIM_HIGH earns the full semantic share.
    # Between the two the share grows in a straight line.
    # These are STARTING VALUES. The team must tune them on its own sample
    # answers; they have not been validated.
    SIM_LOW = 0.30
    SIM_HIGH = 0.80
    SEMANTIC_WEIGHT = 0.70
    KEYWORD_WEIGHT = 0.30
    MARK_STEP = 0.5  # marks are rounded to the nearest half mark
