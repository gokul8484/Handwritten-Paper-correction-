import io
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.evaluator import BagOfWordsSimilarity  # noqa: E402

SCHEME_CSV = (
    "question_no,max_marks,model_answer,keywords\n"
    '1,5,"A primary key uniquely identifies each row in a table and cannot be null.",unique; row; table; null\n'
    '2,5,"Normalization reduces redundancy and improves integrity by splitting tables.",redundancy; integrity; tables\n'
    '3,5,"ACID means atomicity, consistency, isolation and durability.",atomicity; consistency; isolation; durability\n'
)

FAKE_LINES = [
    "Name: Test Student",
    "Q1. a primary key uniquely identifies each row in a table and cannot be null",
    "Q2. normalization reduces redundancy and improves integrity by splitting tables",
    "Q3. ACID means atomicity consistency isolation durability",
]


class FakeEngine:
    """Stands in for TrOCR so tests do not need torch or a model download."""

    def __init__(self, lines=None):
        self.lines = FAKE_LINES if lines is None else lines

    def read_lines(self, images):
        return list(self.lines)


def make_sheet_image(n_lines=4, size=(1600, 1000), angle=0.0):
    """White page with n_lines of dark text (printed font, only for testing)."""
    page = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    for i in range(n_lines):
        y = 150 + i * 180
        cv2.putText(page, f"Q{i + 1}. sample text line number {i + 1}", (120, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 3, cv2.LINE_AA)
    if angle:
        m = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
        page = cv2.warpAffine(page, m, size, borderValue=(255, 255, 255))
    return page


def png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def app(tmp_path):
    # imported here so the module tests can run without the Flask extensions
    from app import create_app
    from models import User, db

    app = create_app({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///" + str(tmp_path / "test.db"),
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "REPORT_FOLDER": str(tmp_path / "reports"),
        "SECRET_KEY": "test",
    })
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)
    app.extensions["ocr_engine"] = FakeEngine()
    app.extensions["similarity"] = BagOfWordsSimilarity()
    with app.app_context():
        db.create_all()
        for name, role in (("admin", "admin"), ("teacher1", "teacher"), ("teacher2", "teacher")):
            u = User(username=name, full_name=name.title(), role=role)
            u.set_password("secret123")
            db.session.add(u)
        db.session.commit()
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def login(client, username="teacher1", password="secret123"):
    return client.post("/login", data={"username": username, "password": password}, follow_redirects=False)


def create_exam(client, csv_text=SCHEME_CSV, title="DBMS Test"):
    return client.post("/exams/new", data={
        "title": title, "subject": "DBMS",
        "scheme": (io.BytesIO(csv_text.encode("utf-8")), "scheme.csv"),
    }, content_type="multipart/form-data", follow_redirects=False)


def upload_sheet(client, exam_id=1, name="Asha", roll="101", data=None, filename="sheet.png"):
    data = png_bytes(make_sheet_image()) if data is None else data
    return client.post(f"/exams/{exam_id}/upload", data={
        "student_name": name, "roll_no": roll,
        "sheet": (io.BytesIO(data), filename),
    }, content_type="multipart/form-data", follow_redirects=False)
