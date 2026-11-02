# Handwritten Exam Evaluator

**AI-Based Handwritten Exam Paper Correction System** - a second-year CSE mini project.

Teachers upload scanned handwritten answer sheets and a model answer key. The system reads the
handwriting with OCR, compares each answer with the model answer using NLP, suggests question-wise
marks, lets the teacher correct them, and produces a result report.

> The marks are **suggestions**. The teacher always reviews and finalises them.
> No accuracy figure is claimed for this project; measure it on your own sample sheets.

## Features

Implemented in this repository:

- Login for teachers and administrators (passwords are hashed)
- Create an exam by uploading a marking scheme (CSV: question number, max marks, model answer, keywords)
- Upload scanned answer sheets as PDF, PNG or JPG
- Image preprocessing: grayscale, background/shadow removal, Otsu threshold, skew correction, ruled-line removal
- Text-line segmentation, then handwriting recognition with **TrOCR** (Tesseract available for neat or printed text)
- Splitting the text into answers using labels such as `Q1.`, `Q2)`, `Ans 3:`
- Answer comparison with **Sentence-BERT** similarity plus keyword matching (word-overlap fallback without torch)
- Mark calculation with a documented formula; question-wise and total marks
- Teacher review: correct recognised text, change marks, add remarks, recalculate, finalise
- PDF report per student and CSV for the whole class
- Admin page to add users and enable or disable accounts

Proposed future work (not implemented): student login and result portal, reading name and roll number
automatically, diagrams and formulas, fine-tuned handwriting model, LLM-based feedback, plagiarism check,
analytics dashboard, cloud deployment.

## Technologies

Python 3.10+, Flask, Flask-Login, Flask-WTF (CSRF), Flask-SQLAlchemy, SQLite, OpenCV, PyMuPDF,
Hugging Face Transformers (TrOCR), Sentence-Transformers (all-MiniLM-L6-v2), ReportLab, pytest.

## Installation

```bash
git clone https://github.com/<your-username>/handwritten-exam-evaluator.git
cd handwritten-exam-evaluator

python -m venv venv
# Windows:  venv\Scripts\activate        Linux / macOS:  source venv/bin/activate

pip install -r requirements.txt          # full: needs internet and 2-3 GB
# or, without torch (web app + tests + word-overlap similarity only):
pip install -r requirements-light.txt
```

The first run downloads the TrOCR and MiniLM models from Hugging Face (internet needed once).

Create the database and the first admin user:

```bash
flask --app app init-db
```

This creates the user `admin` with password `admin123`. **Change it** (Users page, or pass
`--admin-password YourPassword` to the command).

## Run

```bash
flask --app app run
```

Open http://127.0.0.1:5000 and log in.

Useful settings (environment variables):

| Variable | Values | Meaning |
| --- | --- | --- |
| `OCR_ENGINE` | `trocr` (default), `tesseract` | Handwriting model or Tesseract |
| `SIMILARITY_BACKEND` | `sbert` (default), `bow` | Sentence-BERT or word overlap (no torch) |
| `SECRET_KEY` | any long text | Set this for real use |

Example for a computer without torch: `SIMILARITY_BACKEND=bow OCR_ENGINE=tesseract flask --app app run`.
Tesseract also needs the Tesseract program installed on the computer and `pip install pytesseract`.

## Usage (short)

1. Log in. Click **New exam**, enter the title and subject, upload the marking scheme CSV
   (`sample_data/marking_scheme.csv` is an example).
2. Open the exam, click **Upload answer sheet**, enter the student name and roll number, choose the scan.
3. Wait for processing, then check each recognised answer and suggested mark on the review page.
4. Correct text or marks if needed, add remarks, click **Finalise marks**.
5. Download the **PDF report** for the student or the **class CSV** from the exam page.

Writing guidelines for students: write each answer after a label (`Q1.`, `Q2.` ...), use a dark pen on plain
or lightly ruled paper, and leave a clear gap between lines. Scan at 200-300 dpi, upright.

## Try the marking rule without scans

```bash
python evaluate_demo.py sample_data/marking_scheme.csv sample_data/sample_student_answers.txt
python evaluate_demo.py sample_data/marking_scheme.csv sample_data/sample_student_answers.txt --backend sbert
```

## Marking rule

```
semantic_share = clamp((similarity - SIM_LOW) / (SIM_HIGH - SIM_LOW), 0, 1)
score          = 0.7 * semantic_share + 0.3 * keyword_fraction
marks          = round_to_half(score * max_marks)
```

`SIM_LOW = 0.30`, `SIM_HIGH = 0.80`, and the 0.7 / 0.3 weights are starting values in `config.py`.
They have not been validated. Tune them using your own graded sample answers.

## Tests

```bash
pytest -q
```

The tests use a fake OCR engine and the word-overlap similarity, so they check the program logic and
application flow, not handwriting accuracy. Real OCR testing needs your own scanned samples
(see the Testing Report).

## Project structure

```
handwritten-exam-evaluator/
├── app.py                  Flask app: routes, login, review, reports
├── config.py               Settings (OCR engine, thresholds, folders)
├── models.py               Database tables
├── pipeline.py             Scan file -> text -> question-wise marks
├── evaluate_demo.py        Evaluate typed answers from the command line
├── modules/
│   ├── preprocessing.py    Image cleaning and line segmentation
│   ├── ocr.py              TrOCR and Tesseract engines
│   ├── parser.py           Split text into Q1, Q2 ... answers
│   ├── scheme.py           Read the marking-scheme CSV
│   ├── evaluator.py        Similarity, keywords, mark calculation
│   └── report.py           PDF and CSV reports
├── templates/              HTML pages
├── static/css/style.css
├── sample_data/            Example marking scheme and typed answers
├── tests/                  pytest tests
├── requirements.txt
├── requirements-light.txt
└── README.md
```

## Known limitations

English only; best for short descriptive answers; poor scans or untidy handwriting give poor text;
touching text lines may be read wrongly; no diagrams or formulas; slow on a CPU; the 25 MB upload limit;
the default admin password must be changed.

## Team

[Names, roll numbers, guide, college] - fill in before submission.
