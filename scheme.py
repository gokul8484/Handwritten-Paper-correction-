"""Reads the model answers and marking scheme from a CSV file.

Required columns (header row, any order, any letter case):
    question_no, max_marks, model_answer, keywords
Keywords are separated with a semicolon, for example:  unique; row; null
"""
import csv
import io


class SchemeError(ValueError):
    """Raised when the marking-scheme file is not valid. Message is shown to the teacher."""


REQUIRED = ("question_no", "max_marks", "model_answer", "keywords")


def parse_scheme_csv(text):
    """Return a list of dicts: number, max_marks, model_answer, keywords."""
    if text.startswith("﻿"):
        text = text[1:]
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise SchemeError("The marking scheme file is empty.")
    fields = {name.strip().lower(): name for name in reader.fieldnames if name}
    missing = [c for c in REQUIRED if c not in fields]
    if missing:
        raise SchemeError("Missing column(s) in the marking scheme: " + ", ".join(missing))

    questions, seen = [], set()
    for line_no, row in enumerate(reader, start=2):
        if not any((v or "").strip() for v in row.values()):
            continue  # blank line
        try:
            number = int(str(row[fields["question_no"]]).strip())
        except ValueError:
            raise SchemeError(f"Row {line_no}: question_no must be a whole number.")
        try:
            max_marks = float(str(row[fields["max_marks"]]).strip())
        except ValueError:
            raise SchemeError(f"Row {line_no}: max_marks must be a number.")
        if max_marks <= 0:
            raise SchemeError(f"Row {line_no}: max_marks must be greater than 0.")
        answer = (row[fields["model_answer"]] or "").strip()
        if not answer:
            raise SchemeError(f"Row {line_no}: model_answer is empty.")
        if number in seen:
            raise SchemeError(f"Row {line_no}: question {number} appears twice.")
        seen.add(number)
        questions.append({
            "number": number,
            "max_marks": max_marks,
            "model_answer": answer,
            "keywords": (row[fields["keywords"]] or "").strip(),
        })
    if not questions:
        raise SchemeError("The marking scheme has no questions.")
    return sorted(questions, key=lambda q: q["number"])
