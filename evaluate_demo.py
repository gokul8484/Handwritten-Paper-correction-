"""Evaluate TYPED answers against a marking scheme, without any scanned image.

Use it to understand the marking rule, to tune SIM_LOW / SIM_HIGH on your own
sample answers, and as a backup demo when OCR is not available.

    python evaluate_demo.py sample_data/marking_scheme.csv sample_data/sample_student_answers.txt
    python evaluate_demo.py scheme.csv answers.txt --backend sbert
"""
import argparse

from config import Config
from modules import evaluator, parser, scheme


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scheme_csv")
    ap.add_argument("answers_txt")
    ap.add_argument("--backend", default="bow", choices=["bow", "sbert"],
                    help="bow = no downloads (default); sbert = Sentence-BERT")
    args = ap.parse_args()

    cfg = {k: getattr(Config, k) for k in ("SIM_LOW", "SIM_HIGH", "SEMANTIC_WEIGHT", "KEYWORD_WEIGHT", "MARK_STEP")}
    with open(args.scheme_csv, encoding="utf-8-sig") as f:
        questions = scheme.parse_scheme_csv(f.read())
    with open(args.answers_txt, encoding="utf-8") as f:
        answers = parser.split_answers(f.read())
    sim = evaluator.create_similarity(args.backend, Config.SBERT_MODEL)

    total = out_of = 0.0
    for q in questions:
        keywords = [k.strip() for k in q["keywords"].split(";") if k.strip()]
        res = evaluator.evaluate_answer(answers.get(q["number"], ""), q["model_answer"], keywords,
                                        q["max_marks"], sim, cfg)
        total += res["ai_marks"]
        out_of += q["max_marks"]
        print(f"Q{q['number']}: {res['ai_marks']:g}/{q['max_marks']:g}  {res['feedback']}")
    print(f"TOTAL (AI suggested): {total:g}/{out_of:g}")


if __name__ == "__main__":
    main()
