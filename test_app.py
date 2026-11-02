"""End-to-end tests of the web app using the Flask test client.

OCR is replaced by FakeEngine and similarity by the word-overlap back-end,
so these tests check the application flow, not handwriting accuracy.
"""
import numpy as np

from models import AnswerSheet, db
from tests.conftest import (FakeEngine, create_exam, login, make_sheet_image, png_bytes,
                            upload_sheet)


def test_login_logout_and_protected_pages(client):
    assert client.get("/").status_code == 302           # redirected to login
    assert login(client, password="wrong").status_code == 200
    assert login(client).status_code == 302
    assert client.get("/").status_code == 200
    assert client.post("/logout").status_code == 302
    assert client.get("/").status_code == 302


def test_full_flow_upload_review_finalize_reports(client, app):
    login(client)
    assert create_exam(client).status_code == 302
    resp = upload_sheet(client)
    assert resp.status_code == 302 and "/sheets/1" in resp.headers["Location"]

    page = client.get("/sheets/1")
    assert page.status_code == 200 and b"Question 1" in page.data

    with app.app_context():
        sheet = db.session.get(AnswerSheet, 1)
        assert sheet.status == "evaluated" and len(sheet.results) == 3
        assert sheet.total_final > 0
        ids = [r.id for r in sheet.results]

    form = {"action": "finalize"}
    for i in ids:
        form[f"marks_{i}"] = "4"
        form[f"remarks_{i}"] = "ok"
        form[f"text_{i}"] = "corrected text"
    assert client.post("/sheets/1", data=form).status_code == 302
    with app.app_context():
        sheet = db.session.get(AnswerSheet, 1)
        assert sheet.status == "finalized" and sheet.total_final == 12

    pdf = client.get("/sheets/1/report.pdf")
    assert pdf.status_code == 200 and pdf.data.startswith(b"%PDF")
    csv_resp = client.get("/exams/1/results.csv")
    assert b"roll_no,student_name,Q1,Q2,Q3,total,max_total,status" in csv_resp.data
    assert b"101,Asha,4,4,4,12,15,finalized" in csv_resp.data


def test_marks_outside_range_are_rejected(client, app):
    login(client)
    create_exam(client)
    upload_sheet(client)
    with app.app_context():
        rid = db.session.get(AnswerSheet, 1).results[0].id
    client.post("/sheets/1", data={"action": "save", f"marks_{rid}": "99"})
    with app.app_context():
        assert db.session.get(AnswerSheet, 1).results[0].final_marks != 99


def test_recalculate_uses_corrected_text(client, app):
    login(client)
    create_exam(client)
    upload_sheet(client)
    with app.app_context():
        sheet = db.session.get(AnswerSheet, 1)
        form = {"action": "recalculate"}
        for r in sheet.results:
            form[f"text_{r.id}"] = ""           # teacher clears all text
    client.post("/sheets/1", data=form)
    with app.app_context():
        assert db.session.get(AnswerSheet, 1).total_final == 0


# ------------------------------------------------------------ error handling
def test_bad_marking_scheme_is_reported(client):
    login(client)
    resp = create_exam(client, csv_text="question_no,max_marks\n1,5\n")
    assert resp.status_code == 200 and b"Missing column" in resp.data


def test_wrong_file_type_for_sheet(client):
    login(client)
    create_exam(client)
    resp = upload_sheet(client, data=b"hello", filename="notes.txt")
    assert resp.status_code == 200 and b"Only PDF, PNG, JPG" in resp.data


def test_duplicate_roll_number_rejected(client):
    login(client)
    create_exam(client)
    upload_sheet(client)
    resp = upload_sheet(client)
    assert resp.status_code == 200 and b"already uploaded" in resp.data


def test_missing_student_details(client):
    login(client)
    create_exam(client)
    resp = upload_sheet(client, name="", roll="")
    assert resp.status_code == 200 and b"required" in resp.data


def test_corrupt_image_marks_sheet_failed(client, app):
    login(client)
    create_exam(client)
    resp = upload_sheet(client, data=b"this is not a png", filename="bad.png")
    assert resp.status_code == 302
    with app.app_context():
        sheet = db.session.get(AnswerSheet, 1)
        assert sheet.status == "failed" and sheet.error_message


def test_blank_scan_marks_sheet_failed(client, app):
    login(client)
    create_exam(client)
    blank = png_bytes(np.full((1000, 1600, 3), 255, dtype=np.uint8))
    upload_sheet(client, data=blank)
    with app.app_context():
        assert db.session.get(AnswerSheet, 1).status == "failed"


def test_missing_answer_gets_zero(client, app):
    app.extensions["ocr_engine"] = FakeEngine(["Q1. a primary key uniquely identifies each row in a table"])
    login(client)
    create_exam(client)
    upload_sheet(client)
    with app.app_context():
        results = db.session.get(AnswerSheet, 1).results
        assert results[0].ai_marks > 0 and results[1].ai_marks == 0 and results[2].ai_marks == 0


# ------------------------------------------------------------- permissions
def test_teacher_cannot_open_another_teachers_exam(client):
    login(client, "teacher1")
    create_exam(client)
    client.post("/logout")
    login(client, "teacher2")
    assert client.get("/exams/1").status_code == 403
    assert client.get("/exams/1/results.csv").status_code == 403


def test_admin_pages_need_admin_role(client):
    login(client, "teacher1")
    assert client.get("/admin/users").status_code == 403
    client.post("/logout")
    login(client, "admin")
    assert client.get("/admin/users").status_code == 200
    client.post("/admin/users", data={"username": "new", "password": "abcdef", "role": "teacher"})
    assert b"new" in client.get("/admin/users").data
