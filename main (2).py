"""
Image to Excel API
==================
Production-grade table extraction from images.

Engines:
    - pytesseract : primary, fast, runs per-cell with confidence scoring.
    - PaddleOCR   : optional fallback for low-confidence cells / hard images.
                    Loaded lazily so the service starts fine without it.

Pipeline:
    upload -> validate -> preprocess -> deskew -> detect grid lines ->
    detect cells -> dedup (IoU) -> group rows -> OCR per cell w/ confidence ->
    paddle fallback if needed -> clean -> build DataFrame -> coerce types ->
    write xlsx in-memory -> stream back.

Designed for Linux / Docker / Render. Tesseract path defaults to
/usr/bin/tesseract (override with the TESSERACT_CMD env var).
"""

from __future__ import annotations

import io
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("image2excel")

# Tesseract binary path (Linux / Docker / Render)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Per-cell config: PSM 7 = treat as single text line (most cells are 1 line)
OCR_CONFIG_CELL_LINE = "--oem 3 --psm 7"
# Backup config: PSM 6 = uniform block (multi-line cells)
OCR_CONFIG_CELL_BLOCK = "--oem 3 --psm 6"
# Whole-page / fallback config: PSM 4 = single column of text of variable sizes
OCR_CONFIG_PAGE = "--oem 3 --psm 4"

CONF_THRESHOLD = 60.0  # below this avg conf (0-100) we'll try the paddle fallback
PADDLE_ENABLED = os.getenv("PADDLE_ENABLED", "1") not in {"0", "false", "False"}

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/bmp",
    "image/tiff",
    "image/webp",
}

# Cell filtering heuristics
MIN_CELL_W = 18
MIN_CELL_H = 12
MAX_CELL_AREA_RATIO = 0.5          # filter out the outer table border contour
ROW_GROUP_TOLERANCE_FRAC = 0.012   # 1.2% of image height
MIN_CELLS_FOR_TABLE = 4            # below this, switch to layout fallback
DEDUP_IOU_THRESHOLD = 0.6          # 2D IoU; cells above this are duplicates

# ---------------------------------------------------------------------------
# Lazy PaddleOCR wrapper (so the service still starts without paddle installed)
# ---------------------------------------------------------------------------

class _PaddleEngine:
    """Lazy wrapper around PaddleOCR.

    PaddleOCR is heavy (~1 GB models on first run). We only import + initialise
    it when something actually needs the fallback.
    """

    def __init__(self) -> None:
        self._ocr = None
        self._tried = False
        self._available = False

    def _try_init(self) -> None:
        if self._tried:
            return
        self._tried = True
        if not PADDLE_ENABLED:
            log.info("PaddleOCR disabled via PADDLE_ENABLED env var.")
            return
        try:
            from paddleocr import PaddleOCR  # type: ignore
            # show_log silences PaddleOCR's verbose logger
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            self._available = True
            log.info("PaddleOCR initialised (fallback engine ready).")
        except Exception as exc:  # noqa: BLE001
            log.warning("PaddleOCR not available — fallback disabled (%s)", exc)
            self._available = False

    @property
    def available(self) -> bool:
        self._try_init()
        return self._available

    def recognise_cell(self, image: np.ndarray) -> Tuple[str, float]:
        """Recognise text on a small cell crop. Disables detection for speed.

        Returns (text, confidence_0_to_1).
        """
        if not self.available or self._ocr is None or image.size == 0:
            return "", 0.0
        try:
            # cls=True: rotate text if needed; det=False: skip text detection
            result = self._ocr.ocr(image, det=False, cls=True)
        except Exception as exc:  # noqa: BLE001
            log.debug("PaddleOCR cell call failed: %s", exc)
            return "", 0.0
        return _flatten_paddle_recognise(result)

    def full_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run det+rec on a full image; return list of word boxes."""
        if not self.available or self._ocr is None or image.size == 0:
            return []
        try:
            result = self._ocr.ocr(image, cls=True)
        except Exception as exc:  # noqa: BLE001
            log.debug("PaddleOCR full-image call failed: %s", exc)
            return []
        return _flatten_paddle_full(result)


def _flatten_paddle_recognise(result: Any) -> Tuple[str, float]:
    """Pull (text, conf) out of PaddleOCR's nested result for det=False mode."""
    if not result:
        return "", 0.0
    # Result shape varies across paddleocr versions; walk defensively
    items: List[Tuple[str, float]] = []
    for block in result:
        if block is None:
            continue
        # Recognise-only returns [(text, conf)] or [[(text, conf)]]
        if isinstance(block, (list, tuple)) and block and isinstance(block[0], (list, tuple)):
            for entry in block:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    txt, conf = entry[0], entry[1]
                    if isinstance(txt, str):
                        try:
                            items.append((txt, float(conf)))
                        except (TypeError, ValueError):
                            pass
        elif isinstance(block, (list, tuple)) and len(block) >= 2 and isinstance(block[0], str):
            try:
                items.append((block[0], float(block[1])))
            except (TypeError, ValueError):
                pass

    if not items:
        return "", 0.0
    text = " ".join(t for t, _ in items if t)
    avg_conf = sum(c for _, c in items) / len(items)
    return text, avg_conf


def _flatten_paddle_full(result: Any) -> List[Dict[str, Any]]:
    """Normalise PaddleOCR full-image results to [{box, text, conf}]."""
    out: List[Dict[str, Any]] = []
    if not result:
        return out
    for page in result:
        if not page:
            continue
        for entry in page:
            try:
                box, (text, conf) = entry[0], entry[1]
                if not isinstance(text, str) or not text.strip():
                    continue
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                out.append({
                    "text": text,
                    "conf": float(conf),
                    "x": int(min(xs)),
                    "y": int(min(ys)),
                    "w": int(max(xs) - min(xs)),
                    "h": int(max(ys) - min(ys)),
                })
            except (TypeError, ValueError, IndexError):
                continue
    return out


PADDLE = _PaddleEngine()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Image to Excel API",
    version="2.0.0",
    description="High-accuracy image-to-Excel conversion using OpenCV + Tesseract + PaddleOCR.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 1. PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Decode bytes; return (grayscale, inverted_binary).

    Grayscale is used for OCR (more accurate than binary).
    Inverted binary (foreground = white) is used for morphology.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Is the file a valid image?")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Edge-preserving denoise — keeps glyph edges crisp
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=55, sigmaSpace=55)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=10,
    )
    return gray, binary


def deskew_image(gray: np.ndarray, binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Detect skew angle on foreground pixels via minAreaRect; rotate both images.

    No-op when |angle| < 0.5 deg (not worth the resampling) or > 15 deg
    (almost certainly mis-detected — likely a borderless/noisy image).
    """
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 200:
        return gray, binary

    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angle in (-90, 0]; normalise around 0
    if angle < -45:
        angle = 90 + angle
    angle = -angle  # rotation needed to deskew

    if abs(angle) < 0.5 or abs(angle) > 15:
        return gray, binary

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    gray_r = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    bin_r = cv2.warpAffine(
        binary, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    )
    log.info("Deskewed by %.2f deg", angle)
    return gray_r, bin_r


# ---------------------------------------------------------------------------
# 2. TABLE STRUCTURE DETECTION
# ---------------------------------------------------------------------------

def detect_table_lines(binary: np.ndarray) -> np.ndarray:
    """Isolate horizontal + vertical grid lines.

    Kernel sizes scale with image dimensions, so the same code works for
    a 600 px screenshot and a 4000 px scan.
    """
    h, w = binary.shape
    horiz_len = max(20, w // 35)
    vert_len = max(20, h // 35)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    table_mask = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0.0)
    table_mask = cv2.threshold(table_mask, 10, 255, cv2.THRESH_BINARY)[1]

    # Heal small breaks in the grid (faint borders / scan noise)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # Slight dilation so adjoining cell borders fully connect
    table_mask = cv2.dilate(
        table_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1
    )
    return table_mask


# ---------------------------------------------------------------------------
# 3. CELL DETECTION
# ---------------------------------------------------------------------------

def detect_cells(table_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return cell bounding boxes (x, y, w, h) from the grid mask."""
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = table_mask.shape
    img_area = img_h * img_w
    cells: List[Tuple[int, int, int, int]] = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < MIN_CELL_W or h < MIN_CELL_H:
            continue
        if (w * h) / img_area > MAX_CELL_AREA_RATIO:
            continue  # outer table border
        cells.append((x, y, w, h))
    return _dedup_cells_iou(cells)


def _iou_2d(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """2-D IoU between two (x, y, w, h) boxes."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    inter_w = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _iou_x(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """1-D IoU on the x axis (used for in-row dedup)."""
    ax0, _, aw, _ = a
    bx0, _, bw, _ = b
    ax1, bx1 = ax0 + aw, bx0 + bw
    inter = max(0, min(ax1, bx1) - max(ax0, bx0))
    union = max(ax1, bx1) - min(ax0, bx0)
    return inter / union if union else 0.0


def _dedup_cells_iou(cells: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Remove duplicate cells (caused by RETR_TREE returning nested contours)."""
    if not cells:
        return []
    # Sort smallest first so we keep the inner (true) cell over the outer parent
    cells_sorted = sorted(cells, key=lambda c: c[2] * c[3])
    kept: List[Tuple[int, int, int, int]] = []
    for cell in cells_sorted:
        if not any(_iou_2d(cell, k) > DEDUP_IOU_THRESHOLD for k in kept):
            kept.append(cell)
    return kept


# ---------------------------------------------------------------------------
# 4. ROW GROUPING
# ---------------------------------------------------------------------------

def group_rows(
    cells: List[Tuple[int, int, int, int]],
    image_height: int,
) -> List[List[Tuple[int, int, int, int]]]:
    """Cluster cells into rows by similar y; sort cells L→R inside each row."""
    if not cells:
        return []

    tolerance = max(8, int(image_height * ROW_GROUP_TOLERANCE_FRAC))
    cells_sorted = sorted(cells, key=lambda c: c[1])

    rows: List[List[Tuple[int, int, int, int]]] = []
    current: List[Tuple[int, int, int, int]] = [cells_sorted[0]]
    anchor_y = cells_sorted[0][1]

    for cell in cells_sorted[1:]:
        if abs(cell[1] - anchor_y) <= tolerance:
            current.append(cell)
        else:
            rows.append(sorted(current, key=lambda c: c[0]))
            current = [cell]
            anchor_y = cell[1]
    rows.append(sorted(current, key=lambda c: c[0]))

    # In-row dedup on x (same cell discovered as nested contour at same row)
    cleaned: List[List[Tuple[int, int, int, int]]] = []
    for row in rows:
        deduped: List[Tuple[int, int, int, int]] = []
        for cell in row:
            if not any(_iou_x(cell, prev) > 0.7 for prev in deduped):
                deduped.append(cell)
        if deduped:
            cleaned.append(deduped)
    return cleaned


# ---------------------------------------------------------------------------
# 5. OCR + TEXT CLEANING
# ---------------------------------------------------------------------------

_NOISE_RE = re.compile(r"[^\x20-\x7E\u00A0-\uFFFF]")
_MULTI_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip noise, normalise whitespace, trim cell-edge junk."""
    if not text:
        return ""
    text = _NOISE_RE.sub("", text)
    text = _MULTI_WS.sub(" ", text)
    text = text.strip(" |_-—=\t\r\n.,:;")
    return text.strip()


def _tesseract_with_confidence(img: np.ndarray, config: str) -> Tuple[str, float]:
    """Run tesseract and return (joined_text, average_word_confidence_0_to_100)."""
    try:
        data = pytesseract.image_to_data(
            img, config=config, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractError as exc:
        log.debug("Tesseract failed: %s", exc)
        return "", 0.0

    words: List[str] = []
    confs: List[float] = []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        if not txt or not str(txt).strip():
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c < 0:
            continue
        words.append(str(txt))
        confs.append(c)

    text = " ".join(words)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return text, avg_conf


def _preprocess_cell_for_retry(crop: np.ndarray) -> np.ndarray:
    """Threshold + dilate a faint cell crop to thicken strokes for retry OCR."""
    _, b = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    b = cv2.dilate(
        b, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1
    )
    return cv2.bitwise_not(b)  # back to dark text on light background


def extract_text_from_cell(gray: np.ndarray, box: Tuple[int, int, int, int]) -> str:
    """OCR a single cell with multi-pass strategy and confidence-based fallback.

    Pass 1: tesseract PSM 7 on the grayscale crop (single line).
    Pass 2: tesseract PSM 6 on a thresholded + dilated crop (block of text).
    Pass 3: PaddleOCR on the original crop (only if available and conf is low).
    """
    x, y, w, h = box
    pad = 3
    x0 = max(0, x + pad)
    y0 = max(0, y + pad)
    x1 = max(x0 + 1, x + w - pad)
    y1 = max(y0 + 1, y + h - pad)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape) < 6:
        return ""

    # Pass 1
    text1, conf1 = _tesseract_with_confidence(crop, OCR_CONFIG_CELL_LINE)
    text1 = clean_text(text1)
    if text1 and conf1 >= CONF_THRESHOLD:
        return text1

    # Pass 2
    crop_proc = _preprocess_cell_for_retry(crop)
    text2, conf2 = _tesseract_with_confidence(crop_proc, OCR_CONFIG_CELL_BLOCK)
    text2 = clean_text(text2)

    # Best of 1 and 2
    best_text, best_conf = (text1, conf1) if conf1 >= conf2 else (text2, conf2)

    # Pass 3 — PaddleOCR fallback when confidence is still low
    if (not best_text or best_conf < CONF_THRESHOLD) and PADDLE.available:
        # Paddle prefers BGR; convert grayscale crop
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        ptext, pconf = PADDLE.recognise_cell(crop_bgr)
        ptext = clean_text(ptext)
        # paddle conf is 0-1; tesseract conf is 0-100
        pconf_pct = pconf * 100.0
        if ptext and pconf_pct > best_conf:
            return ptext

    return best_text


# ---------------------------------------------------------------------------
# 6. LAYOUT-AWARE FALLBACK PARSER
# ---------------------------------------------------------------------------

def fallback_parser(gray: np.ndarray) -> List[List[str]]:
    """Reconstruct a table from full-image OCR using word bounding boxes.

    Strategy:
        1. Get word-level boxes from tesseract image_to_data.
        2. (Optionally) merge with PaddleOCR boxes for better recall.
        3. Cluster words into lines (same line_num).
        4. Within each line, split on horizontal gaps significantly larger
           than the local character width — those are column separators.
        5. Pad each row to the max column count.
    """
    words: List[Dict[str, Any]] = _tesseract_words(gray)

    # Augment with paddle if it's available and tesseract found very few words
    if PADDLE.available and len(words) < 8:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        paddle_words = PADDLE.full_image(bgr)
        # Tag with synthetic line_num based on y-band
        for pw in paddle_words:
            pw["line_num"] = pw["y"] // max(1, pw["h"])
            pw["block_num"] = 0
            pw["par_num"] = 0
        # Use whichever set has more words
        if len(paddle_words) > len(words):
            words = paddle_words

    if not words:
        return []

    # Group words into lines
    line_groups: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for w in words:
        key = (w.get("block_num", 0), w.get("par_num", 0), w.get("line_num", 0))
        line_groups[key].append(w)

    lines: List[List[Dict[str, Any]]] = []
    for ws in line_groups.values():
        ws.sort(key=lambda w: w["x"])
        lines.append(ws)
    lines.sort(key=lambda ws: sum(w["y"] for w in ws) / len(ws))

    # Split each line into columns by x-gaps
    rows: List[List[str]] = []
    for line_words in lines:
        if not line_words:
            continue
        groups: List[List[Dict[str, Any]]] = [[line_words[0]]]
        for prev, curr in zip(line_words, line_words[1:]):
            gap = curr["x"] - (prev["x"] + prev["w"])
            # Local char-width estimate from the previous word
            est_char_w = max(prev["h"] * 0.5, 5)
            if gap > est_char_w * 2.0:  # column break
                groups.append([curr])
            else:
                groups[-1].append(curr)

        row = [clean_text(" ".join(w["text"] for w in g)) for g in groups]
        row = [c for c in row if c]
        if row:
            rows.append(row)

    return rows


def _tesseract_words(gray: np.ndarray) -> List[Dict[str, Any]]:
    """Get word-level bounding boxes from tesseract."""
    try:
        data = pytesseract.image_to_data(
            gray, config=OCR_CONFIG_PAGE, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractError as exc:
        log.warning("Fallback tesseract image_to_data failed: %s", exc)
        return []

    words: List[Dict[str, Any]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < 0:
            continue
        words.append({
            "text": text,
            "conf": conf,
            "x": int(data["left"][i]),
            "y": int(data["top"][i]),
            "w": int(data["width"][i]),
            "h": int(data["height"][i]),
            "line_num": int(data["line_num"][i]),
            "block_num": int(data["block_num"][i]),
            "par_num": int(data["par_num"][i]),
        })
    return words


# ---------------------------------------------------------------------------
# 7. DATAFRAME ASSEMBLY + TYPE COERCION
# ---------------------------------------------------------------------------

# Conservative numeric OCR fixes — applied only when a column looks numeric
_OCR_NUM_FIXES = str.maketrans({
    "O": "0", "o": "0",
    "l": "1", "I": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
})

_CURRENCY_CHARS = "$€£¥₹"


def _looks_numeric(s: str) -> bool:
    """True if `s` could be parsed as a number after stripping currency/commas."""
    if not s:
        return False
    cleaned = s
    for ch in _CURRENCY_CHARS:
        cleaned = cleaned.replace(ch, "")
    cleaned = cleaned.replace(",", "").replace(" ", "").replace("%", "").strip()
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _detect_header(rows: List[List[str]]) -> bool:
    """Decide whether the first row should become column headers.

    Heuristics:
      - all cells non-empty, AND
      - mostly non-numeric (numeric headers are rare).
    """
    if not rows or len(rows) < 2:
        return False
    first = rows[0]
    if not all(cell.strip() for cell in first):
        return False
    numeric = sum(1 for c in first if _looks_numeric(c))
    return numeric <= len(first) * 0.3


def _coerce_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Try numeric, then date conversion per column.

    Apply only when at least 70% of non-empty cells parse successfully.
    """
    for col in df.columns:
        series = df[col].astype(str)
        non_empty = series.str.strip().ne("").sum()
        if not non_empty:
            continue

        # ---- numeric (with OCR digit-look-alike fixes) ----
        cleaned = (
            series.str.translate(_OCR_NUM_FIXES)
            .str.replace(",", "", regex=False)
            .str.replace(r"[$€£¥₹%\s]", "", regex=True)
            .str.strip()
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().sum() / non_empty >= 0.7:
            df[col] = numeric
            continue

        # ---- date ----
        dates = pd.to_datetime(series, errors="coerce", dayfirst=False)
        if dates.notna().sum() / non_empty >= 0.7:
            df[col] = dates

    return df


def build_dataframe(
    rows_of_cells: List[List[Tuple[int, int, int, int]]],
    gray: np.ndarray,
) -> pd.DataFrame:
    """Run OCR over every detected cell and assemble a DataFrame."""
    extracted: List[List[str]] = []
    for row in rows_of_cells:
        extracted.append([extract_text_from_cell(gray, box) for box in row])
    return _finalize_dataframe(extracted)


def _finalize_dataframe(rows: List[List[str]]) -> pd.DataFrame:
    """Common path for both grid pipeline and layout fallback."""
    if not rows:
        return pd.DataFrame()

    # Pad to rectangular shape — fixes misaligned rows
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    if _detect_header(rows):
        header = rows[0]
        body = rows[1:]
        # Deduplicate header names
        seen: Dict[str, int] = {}
        unique_header: List[str] = []
        for col in header:
            base = col.strip() or "Column"
            if base in seen:
                seen[base] += 1
                unique_header.append(f"{base}_{seen[base]}")
            else:
                seen[base] = 0
                unique_header.append(base)
        df = pd.DataFrame(body, columns=unique_header)
    else:
        df = pd.DataFrame(rows, columns=[f"Column_{i + 1}" for i in range(max_cols)])

    # Drop rows / columns that are entirely empty or whitespace
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.fillna("")

    df = _coerce_column_types(df)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# 8. EXCEL EXPORT
# ---------------------------------------------------------------------------

def dataframe_to_excel_bytes(df: pd.DataFrame) -> io.BytesIO:
    """Serialise DataFrame to an in-memory .xlsx with auto-sized columns."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        sheet = "Sheet1"
        if df.empty:
            pd.DataFrame({"info": ["No table content extracted from image."]}).to_excel(
                writer, sheet_name=sheet, index=False
            )
        else:
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            for idx, column in enumerate(df.columns, start=1):
                values = [str(column)] + [str(v) for v in df.iloc[:, idx - 1].tolist()]
                width = min(max(len(v) for v in values) + 2, 60)
                ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = width
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# 9. PIPELINE ORCHESTRATION
# ---------------------------------------------------------------------------

def image_bytes_to_dataframe(image_bytes: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the full pipeline and return (DataFrame, debug_metrics)."""
    gray, binary = preprocess_image(image_bytes)
    gray, binary = deskew_image(gray, binary)

    table_mask = detect_table_lines(binary)
    cells = detect_cells(table_mask)
    log.info("Detected %d candidate cells", len(cells))

    if len(cells) < MIN_CELLS_FOR_TABLE:
        log.info("Few cells detected — switching to layout fallback")
        rows_text = fallback_parser(gray)
        df = _finalize_dataframe(rows_text)
        return df, {"mode": "fallback_layout", "cells": len(cells), "rows": len(rows_text)}

    rows = group_rows(cells, image_height=gray.shape[0])
    log.info("Grouped into %d rows", len(rows))

    df = build_dataframe(rows, gray)
    if df.empty:
        log.info("Grid OCR produced empty result — retrying with layout fallback")
        rows_text = fallback_parser(gray)
        df = _finalize_dataframe(rows_text)
        return df, {"mode": "fallback_after_empty_grid", "cells": len(cells)}

    return df, {"mode": "grid", "cells": len(cells), "rows": len(rows)}


# ---------------------------------------------------------------------------
# 10. ROUTES
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _startup() -> None:
    try:
        ver = pytesseract.get_tesseract_version()
        log.info("Tesseract %s ready at %s", ver, TESSERACT_CMD)
    except Exception as exc:  # pragma: no cover
        log.error("Tesseract not callable at %s: %s", TESSERACT_CMD, exc)
    if PADDLE_ENABLED:
        log.info("PaddleOCR will be initialised lazily on first fallback need.")


@app.get("/")
def root() -> dict:
    return {"service": "image-to-excel", "version": app.version, "status": "ok"}


@app.get("/health")
def health() -> dict:
    try:
        ver = str(pytesseract.get_tesseract_version())
    except Exception:
        ver = "unavailable"
    return {
        "status": "ok",
        "tesseract": ver,
        "paddle_enabled": PADDLE_ENABLED,
        "paddle_loaded": PADDLE._available,
    }


@app.post("/convert")
async def convert(file: UploadFile = File(...)) -> StreamingResponse:
    # ---- validate ----------------------------------------------------------
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {file.content_type}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded.",
        )
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    # ---- pipeline ----------------------------------------------------------
    try:
        df, metrics = image_bytes_to_dataframe(contents)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)
        ) from ve
    except pytesseract.TesseractNotFoundError as tnf:
        log.exception("Tesseract binary missing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OCR engine not available on server.",
        ) from tnf
    except Exception as exc:  # noqa: BLE001
        log.exception("Conversion failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error processing image: {exc.__class__.__name__}",
        ) from exc

    # ---- write xlsx --------------------------------------------------------
    try:
        excel_buf = dataframe_to_excel_bytes(df)
    except Exception as exc:  # noqa: BLE001
        log.exception("Excel serialisation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build Excel file.",
        ) from exc

    out_name = (os.path.splitext(file.filename or "table")[0] or "table") + ".xlsx"
    headers = {
        "Content-Disposition": f'attachment; filename="{out_name}"',
        "X-Extraction-Mode": str(metrics.get("mode", "unknown")),
        "X-Detected-Cells": str(metrics.get("cells", 0)),
        "X-Rows": str(len(df)),
        "X-Columns": str(len(df.columns)),
    }
    return StreamingResponse(
        excel_buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@app.exception_handler(Exception)
async def _unhandled(_, exc: Exception) -> JSONResponse:  # pragma: no cover
    log.exception("Unhandled: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ---------------------------------------------------------------------------
# 11. ENTRYPOINT (Render uses $PORT)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
        reload=False,
    )
