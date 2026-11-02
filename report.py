"""Module 8: Report generation (PDF for one student, CSV for the whole class)."""
import csv
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _fmt(value):
    value = float(value or 0)
    return str(int(value)) if value == int(value) else f"{value:g}"


def build_student_pdf(path, exam, sheet):
    """Write a one-student result report to `path` (PDF)."""
    styles = getSampleStyleSheet()
    small = styles["BodyText"].clone("small")
    small.fontSize = 8.5
    small.leading = 11

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title=f"Result - {sheet.student_name}")
    story = [
        Paragraph("Answer Sheet Evaluation Report", styles["Title"]),
        Paragraph(f"<b>Exam:</b> {escape(exam.title)} &nbsp;&nbsp; <b>Subject:</b> {escape(exam.subject or '-')}",
                  styles["BodyText"]),
        Paragraph(f"<b>Student:</b> {escape(sheet.student_name)} &nbsp;&nbsp; <b>Roll No:</b> {escape(sheet.roll_no)}",
                  styles["BodyText"]),
        Spacer(1, 0.4 * cm),
    ]

    rows = [["Q.No", "Max", "AI marks", "Final marks", "Teacher remark / system note"]]
    for r in sheet.results:
        note = r.remarks.strip() if (r.remarks or "").strip() else (r.feedback or "")
        rows.append([str(r.question_no), _fmt(r.question.max_marks), _fmt(r.ai_marks),
                     _fmt(r.final_marks), Paragraph(escape(note), small)])
    rows.append(["Total", _fmt(exam.total_marks), _fmt(sheet.total_ai), _fmt(sheet.total_final), ""])

    table = Table(rows, colWidths=[1.4 * cm, 1.4 * cm, 2.2 * cm, 2.4 * cm, 9.6 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e8eef6")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (3, -1), "CENTER"),
    ]))
    story += [table, Spacer(1, 0.6 * cm)]

    status = "Finalised by teacher" if sheet.status == "finalized" else "DRAFT - not yet finalised by the teacher"
    story += [
        Paragraph(f"<b>Total: {_fmt(sheet.total_final)} / {_fmt(exam.total_marks)}</b>", styles["Heading2"]),
        Paragraph(f"Status: {status}", styles["BodyText"]),
        Paragraph("AI marks are suggestions made by the system. Final marks are the marks approved by the teacher.",
                  small),
        Paragraph(f"Generated on {datetime.now():%d %b %Y, %H:%M}", small),
    ]
    doc.build(story)
    return str(path)


def build_class_csv(path, exam):
    """Write one row per student with question-wise final marks."""
    numbers = [q.number for q in exam.questions]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["roll_no", "student_name"] + [f"Q{n}" for n in numbers]
                        + ["total", "max_total", "status"])
        for sheet in exam.sheets:
            by_q = {r.question_no: r.final_marks for r in sheet.results}
            writer.writerow([sheet.roll_no, sheet.student_name]
                            + [_fmt(by_q.get(n, 0)) for n in numbers]
                            + [_fmt(sheet.total_final), _fmt(exam.total_marks), sheet.status])
    return str(path)
