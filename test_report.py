"""Report generation tests. Plain Python objects stand in for database rows."""
from types import SimpleNamespace as NS

from modules import report


def _exam_and_sheet(status="finalized"):
    q1, q2 = NS(number=1, max_marks=5.0), NS(number=2, max_marks=5.0)
    results = [
        NS(question_no=1, question=q1, ai_marks=4.0, final_marks=4.5, remarks="Good <answer> & clear", feedback=""),
        NS(question_no=2, question=q2, ai_marks=2.0, final_marks=2.0, remarks="", feedback="Meaning similarity 0.40."),
    ]
    sheet = NS(student_name="Asha", roll_no="101", status=status, results=results,
               total_final=6.5, total_ai=6.0)
    exam = NS(title="DBMS Test", subject="DBMS", total_marks=10.0, questions=[q1, q2], sheets=[sheet])
    return exam, sheet


def test_student_pdf_is_created(tmp_path):
    exam, sheet = _exam_and_sheet()
    path = report.build_student_pdf(tmp_path / "r.pdf", exam, sheet)
    data = open(path, "rb").read()
    assert data.startswith(b"%PDF") and len(data) > 1000


def test_draft_report_also_builds(tmp_path):
    exam, sheet = _exam_and_sheet(status="evaluated")
    assert report.build_student_pdf(tmp_path / "d.pdf", exam, sheet)


def test_class_csv_content(tmp_path):
    exam, _ = _exam_and_sheet()
    path = report.build_class_csv(tmp_path / "c.csv", exam)
    lines = open(path, encoding="utf-8").read().splitlines()
    assert lines[0] == "roll_no,student_name,Q1,Q2,total,max_total,status"
    assert lines[1] == "101,Asha,4.5,2,6.5,10,finalized"
