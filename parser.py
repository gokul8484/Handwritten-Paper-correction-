"""Module 4: Split the recognised text into question-wise answers.

The student must write a label before each answer, for example
    Q1. ...      Q 2) ...      Question 3: ...      Ans 4 - ...
Text before the first label (name, date, header) is ignored.
"""
import re
from collections import OrderedDict

# OCR often reads the digit 1 as I, l or | and 0 as O.
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"})

_MARKER = re.compile(
    r"^\s*(?:q(?:uestion)?|ans(?:wer)?)\s*[.\-:]?\s*([0-9OoIl|]{1,2})\s*[).:\-]?\s*(.*)$",
    re.IGNORECASE,
)


def _to_number(token):
    cleaned = token.translate(_DIGIT_FIX)
    return int(cleaned) if cleaned.isdigit() else None


def split_answers(text):
    """Return OrderedDict {question_number: answer_text}."""
    answers = OrderedDict()
    current = None
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _MARKER.match(line)
        if match:
            number = _to_number(match.group(1))
            if number is not None:
                current = number
                answers.setdefault(current, [])
                rest = match.group(2).strip()
                if rest:
                    answers[current].append(rest)
                continue
        if current is not None:
            answers[current].append(line)
    return OrderedDict((n, " ".join(parts).strip()) for n, parts in answers.items())
