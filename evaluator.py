"""Modules 5 and 6: Answer comparison and mark calculation.

Marking rule (simple and explainable):

    semantic_share = clamp((similarity - SIM_LOW) / (SIM_HIGH - SIM_LOW), 0, 1)
    score          = SEMANTIC_WEIGHT * semantic_share + KEYWORD_WEIGHT * keyword_fraction
    ai_marks       = round_to_half(score * max_marks)

If a question has no keywords, the whole score comes from the semantic share.
An empty answer always gets 0.
"""
import math
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "at",
    "and", "or", "it", "its", "that", "this", "these", "those", "for", "with", "as", "by",
    "from", "which", "can", "will", "has", "have", "had", "so", "such", "than", "then",
}


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def stem(word):
    """Very small stemmer so that 'tables', 'table' and 'tabled' all match.

    Steps: plural ending, then -ing / -ed / -ly, then a trailing silent 'e'.
    It is crude on purpose (a student project, not a linguistics library).
    """
    # 1. plurals
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith(("sses", "xes", "ches", "shes", "zes")):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        word = word[:-1]
    # 2. verb / adverb endings
    for suffix in ("ing", "ed", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    # 3. silent e
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def stems(text):
    return [stem(t) for t in tokenize(text)]


# --------------------------------------------------------------------------
# Keyword matching
# --------------------------------------------------------------------------
def keyword_match(answer, keywords):
    """Return (fraction_matched, matched_list, missing_list)."""
    if not keywords:
        return 0.0, [], []
    answer_stems = set(stems(answer))
    matched, missing = [], []
    for kw in keywords:
        kw_stems = stems(kw)
        if kw_stems and all(s in answer_stems for s in kw_stems):
            matched.append(kw)
        else:
            missing.append(kw)
    return len(matched) / len(keywords), matched, missing


# --------------------------------------------------------------------------
# Similarity back-ends (same method name: similarity(a, b) -> 0..1)
# --------------------------------------------------------------------------
class BagOfWordsSimilarity:
    """Simple word-overlap cosine similarity. No downloads needed.

    It cannot understand synonyms, so it is weaker than Sentence-BERT.
    Use it for quick tests or on a computer that cannot run the neural models.
    """

    def similarity(self, a, b):
        ca = Counter(s for s in stems(a) if s not in STOPWORDS)
        cb = Counter(s for s in stems(b) if s not in STOPWORDS)
        if not ca or not cb:
            return 0.0
        dot = sum(ca[w] * cb[w] for w in ca)
        norm = math.sqrt(sum(v * v for v in ca.values())) * math.sqrt(sum(v * v for v in cb.values()))
        return dot / norm if norm else 0.0


class SbertSimilarity:
    """Cosine similarity of Sentence-BERT embeddings (all-MiniLM-L6-v2 by default)."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer, util

        self._util = util
        self.model = SentenceTransformer(model_name)

    def similarity(self, a, b):
        if not (a or "").strip() or not (b or "").strip():
            return 0.0
        emb = self.model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        value = float(self._util.cos_sim(emb[0], emb[1]).item())
        return max(0.0, min(1.0, value))


def create_similarity(backend, sbert_model="sentence-transformers/all-MiniLM-L6-v2"):
    backend = (backend or "").lower()
    if backend == "sbert":
        return SbertSimilarity(sbert_model)
    if backend == "bow":
        return BagOfWordsSimilarity()
    raise ValueError(f"Unknown similarity backend '{backend}'. Use 'sbert' or 'bow'.")


# --------------------------------------------------------------------------
# Marks
# --------------------------------------------------------------------------
def semantic_share(similarity, low, high):
    if high <= low:
        raise ValueError("SIM_HIGH must be greater than SIM_LOW")
    return max(0.0, min(1.0, (similarity - low) / (high - low)))


def round_to_step(value, step):
    return round(value / step) * step


def calculate_marks(similarity, keyword_fraction, has_keywords, max_marks, cfg):
    """Apply the marking rule and return marks between 0 and max_marks."""
    share = semantic_share(similarity, cfg["SIM_LOW"], cfg["SIM_HIGH"])
    if has_keywords:
        score = cfg["SEMANTIC_WEIGHT"] * share + cfg["KEYWORD_WEIGHT"] * keyword_fraction
    else:
        score = share
    marks = round_to_step(score * max_marks, cfg["MARK_STEP"])
    return max(0.0, min(float(max_marks), marks))


def evaluate_answer(answer, model_answer, keywords, max_marks, similarity_model, cfg):
    """Evaluate one student answer. Returns a dict that is easy to store."""
    answer = (answer or "").strip()
    if not answer:
        return {
            "similarity": 0.0, "keyword_score": 0.0, "ai_marks": 0.0,
            "matched": [], "missing": list(keywords),
            "feedback": "No answer was detected for this question.",
        }
    sim = similarity_model.similarity(answer, model_answer)
    kw_fraction, matched, missing = keyword_match(answer, keywords)
    marks = calculate_marks(sim, kw_fraction, bool(keywords), max_marks, cfg)

    parts = [f"Meaning similarity {sim:.2f}."]
    if keywords:
        parts.append(f"Keywords matched {len(matched)} of {len(keywords)}.")
        if missing:
            parts.append("Missing: " + ", ".join(missing) + ".")
    return {
        "similarity": round(sim, 4), "keyword_score": round(kw_fraction, 4),
        "ai_marks": marks, "matched": matched, "missing": missing,
        "feedback": " ".join(parts),
    }
