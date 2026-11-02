"""Module 2: Image preprocessing and text-line segmentation.

Steps for every page:
  1. load the page (image or PDF page) and convert to grayscale
  2. resize to a standard width
  3. remove uneven background / shadows
  4. binarise (Otsu threshold)
  5. estimate and correct page skew
  6. remove long ruled lines
  7. cut the page into single text lines (horizontal projection profile)
"""
from pathlib import Path

import cv2
import numpy as np

TARGET_WIDTH = 1600


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_pages(path, dpi=200):
    """Return a list of BGR images, one per page."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        import fitz  # PyMuPDF

        pages = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if not pages:
            raise ValueError("The PDF file has no pages.")
        return pages

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError("The image file could not be read. It may be damaged or not an image.")
    return [img]


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def resize_to_width(gray, width=TARGET_WIDTH):
    h, w = gray.shape[:2]
    if w == width:
        return gray
    scale = width / float(w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, (width, max(1, int(round(h * scale)))), interpolation=interp)


def normalize_background(gray):
    """Divide by an estimate of the paper background to remove shadows."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), 15)
    background = np.maximum(background, 1)
    return cv2.divide(gray, background, scale=255)


def binarize(gray):
    """Otsu threshold. Ink = 0 (black), paper = 255 (white)."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _rotate(img, angle, border_value):
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


def estimate_skew(binary, max_angle=8.0, step=0.5):
    """Find the rotation (degrees) that makes text rows most distinct.

    The correct angle is the one where the row-wise ink profile has the
    largest variance (text rows and white gaps are clearly separated).
    """
    ink = (binary == 0).astype(np.uint8) * 255
    small = cv2.resize(ink, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    if small.sum() == 0:
        return 0.0
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 1e-9, step):
        rotated = _rotate(small, float(angle), 0)
        score = float(np.var(rotated.sum(axis=1, dtype=np.float64)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def remove_ruled_lines(binary):
    """Remove long horizontal lines (notebook rules, underlines)."""
    ink = 255 - binary
    length = max(40, binary.shape[1] // 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    cleaned = cv2.subtract(ink, lines)
    return 255 - cleaned


def preprocess_page(img):
    """Return (clean_gray, binary, skew_angle) for one page image."""
    gray = resize_to_width(to_gray(img))
    clean = normalize_background(gray)
    binary = binarize(clean)
    angle = estimate_skew(binary)
    if abs(angle) >= 0.3:
        clean = _rotate(clean, angle, 255)
        binary = binarize(clean)
    binary = remove_ruled_lines(binary)
    return clean, binary, angle


# --------------------------------------------------------------------------
# Line segmentation
# --------------------------------------------------------------------------
def _find_runs(mask):
    """Return [(start, end)] for every run of True values in a 1-D array."""
    runs, start = [], None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def segment_lines(clean_gray, binary, min_height=14, merge_gap=8):
    """Cut a page into text-line images (grayscale crops).

    Works best on pages with one answer line per row and clear gaps
    between rows. Touching or overlapping lines are a known limitation.
    """
    ink = (binary == 0).astype(np.uint8)
    h, w = ink.shape
    profile = ink.sum(axis=1).astype(np.float64)
    k = max(3, h // 200)
    smooth = np.convolve(profile, np.ones(k) / k, mode="same")
    threshold = max(2.0, 0.006 * w)
    runs = _find_runs(smooth > threshold)

    # merge runs separated by a very small gap (dots on i, accents, etc.)
    merged = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    merged = [(s, e) for s, e in merged if e - s >= min_height]
    if not merged:
        return []

    # split very tall runs (two lines glued together) into equal parts
    median_h = float(np.median([e - s for s, e in merged]))
    final = []
    for s, e in merged:
        parts = int(round((e - s) / median_h)) if (e - s) > 2.2 * median_h else 1
        if parts <= 1:
            final.append((s, e))
            continue
        size = (e - s) / parts
        for p in range(parts):
            final.append((int(s + p * size), int(s + (p + 1) * size)))

    crops = []
    for s, e in final:
        pad = int(0.25 * (e - s)) + 2
        y0, y1 = max(0, s - pad), min(h, e + pad)
        band = ink[y0:y1]
        cols = np.where(band.sum(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        x0, x1 = max(0, cols[0] - 12), min(w, cols[-1] + 12)
        if x1 - x0 < 20:
            continue
        crops.append(clean_gray[y0:y1, x0:x1])
    return crops
