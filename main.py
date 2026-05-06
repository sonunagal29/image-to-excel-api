from __future__ import annotations

import io
import logging
import os
import re
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("image2excel")

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff", "image/webp"}

app = FastAPI(title="Image to Excel API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Extraction-Mode", "X-Rows", "X-Columns"],
)

_CLEAN_RE = re.compile(r"[^\x20-\x7E\u00A0-\uFFFF]")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _CLEAN_RE.sub("", text or "")
    text = _WS_RE.sub(" ", text)
    return text.strip(" |_-—=\t\r\n")


def decode_and_prepare(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")

    h, w = img.shape[:2]
    # OCR fails on small screenshots. Upscale before OCR.
    target_w = 1800
    if w < target_w:
        scale = min(4.0, target_w / max(w, 1))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Improve contrast without destroying table borders.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def ocr_words(gray: np.ndarray) -> List[Dict[str, Any]]:
    configs = [
        "--oem 3 --psm 6 -l eng",
        "--oem 3 --psm 11 -l eng",
    ]
    best: List[Dict[str, Any]] = []

    for config in configs:
        data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
        words: List[Dict[str, Any]] = []
        for i, txt in enumerate(data.get("text", [])):
            txt = clean_text(str(txt))
            if not txt:
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1
            if conf < 25:
                continue
            x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
            if w <= 0 or h <= 0:
                continue
            words.append({"text": txt, "conf": conf, "x": x, "y": y, "w": w, "h": h, "cx": x + w / 2, "cy": y + h / 2})
        if len(words) > len(best):
            best = words
    return best


def cluster_rows(words: List[Dict[str, Any]], image_h: int) -> List[List[Dict[str, Any]]]:
    if not words:
        return []
    heights = [w["h"] for w in words]
    row_tol = max(12, int(np.median(heights) * 0.75), int(image_h * 0.008))
    words_sorted = sorted(words, key=lambda w: w["cy"])

    rows: List[List[Dict[str, Any]]] = []
    for word in words_sorted:
        placed = False
        for row in rows:
            avg_y = sum(w["cy"] for w in row) / len(row)
            if abs(word["cy"] - avg_y) <= row_tol:
                row.append(word)
                placed = True
                break
        if not placed:
            rows.append([word])

    rows = [sorted(row, key=lambda w: w["x"]) for row in rows]
    rows.sort(key=lambda row: sum(w["cy"] for w in row) / len(row))
    return rows


def split_row_into_cells(row: List[Dict[str, Any]]) -> List[str]:
    if not row:
        return []
    row = sorted(row, key=lambda w: w["x"])
    heights = [w["h"] for w in row]
    median_h = float(np.median(heights)) if heights else 12.0
    groups: List[List[Dict[str, Any]]] = [[row[0]]]

    for prev, cur in zip(row, row[1:]):
        gap = cur["x"] - (prev["x"] + prev["w"])
        prev_char_w = max(prev["w"] / max(len(prev["text"]), 1), median_h * 0.45, 5)
        # Large gap = next table column. Normal word gap remains same cell.
        threshold = max(18, prev_char_w * 2.8, median_h * 0.9)
        if gap > threshold:
            groups.append([cur])
        else:
            groups[-1].append(cur)

    cells = [clean_text(" ".join(w["text"] for w in group)) for group in groups]
    return [c for c in cells if c]


def build_rows_from_words(gray: np.ndarray) -> List[List[str]]:
    words = ocr_words(gray)
    log.info("OCR words found: %d", len(words))
    rows_words = cluster_rows(words, gray.shape[0])
    rows = [split_row_into_cells(row) for row in rows_words]
    rows = [r for r in rows if r]

    # Remove obvious browser/website noise if a full screenshot is uploaded.
    # Keep the densest table-like block: consecutive rows with 2+ cells.
    tableish = [i for i, r in enumerate(rows) if len(r) >= 2]
    if tableish:
        blocks: List[List[int]] = []
        cur = [tableish[0]]
        for idx in tableish[1:]:
            if idx == cur[-1] + 1:
                cur.append(idx)
            else:
                blocks.append(cur)
                cur = [idx]
        blocks.append(cur)
        best_block = max(blocks, key=len)
        if len(best_block) >= 2:
            start, end = best_block[0], best_block[-1]
            rows = rows[start : end + 1]
    return rows


def detect_grid_cells(gray: np.ndarray) -> List[List[Tuple[int, int, int, int]]]:
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    h, w = binary.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 40), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 40)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    mask = cv2.add(horizontal, vertical)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    img_area = h * w
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if cw < 35 or ch < 18:
            continue
        if area > img_area * 0.45:
            continue
        boxes.append((x, y, cw, ch))

    # Deduplicate nested boxes.
    boxes = sorted(boxes, key=lambda b: b[2] * b[3])
    kept: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        if not any(iou(b, k) > 0.7 for k in kept):
            kept.append(b)

    if len(kept) < 6:
        return []

    # Group by row.
    kept.sort(key=lambda b: b[1] + b[3] / 2)
    tol = max(12, int(np.median([b[3] for b in kept]) * 0.6))
    rows: List[List[Tuple[int, int, int, int]]] = []
    for b in kept:
        cy = b[1] + b[3] / 2
        for row in rows:
            avg = sum(x[1] + x[3] / 2 for x in row) / len(row)
            if abs(cy - avg) <= tol:
                row.append(b)
                break
        else:
            rows.append([b])
    rows = [sorted(r, key=lambda b: b[0]) for r in rows if len(r) >= 2]
    rows.sort(key=lambda r: sum(b[1] for b in r) / len(r))
    return rows


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    iw = max(0, min(ax2, bx2) - max(ax, bx))
    ih = max(0, min(ay2, by2) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def ocr_cell(gray: np.ndarray, box: Tuple[int, int, int, int]) -> str:
    x, y, w, h = box
    pad = max(2, min(w, h) // 12)
    crop = gray[max(0, y + pad) : max(0, y + h - pad), max(0, x + pad) : max(0, x + w - pad)]
    if crop.size == 0:
        return ""
    crop = cv2.resize(crop, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(crop, config="--oem 3 --psm 7 -l eng")
    return clean_text(text)


def rows_from_grid(gray: np.ndarray) -> List[List[str]]:
    grid = detect_grid_cells(gray)
    if not grid:
        return []
    rows = [[ocr_cell(gray, box) for box in row] for row in grid]
    rows = [[c for c in r] for r in rows if any(clean_text(c) for c in r)]
    return rows


def finalize_dataframe(rows: List[List[str]]) -> pd.DataFrame:
    rows = [[clean_text(c) for c in row] for row in rows]
    rows = [row for row in rows if any(row)]
    if not rows:
        return pd.DataFrame({"info": ["No readable table content extracted from image."]})

    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    # Header detection: first row mostly text and non-empty.
    first = rows[0]
    non_empty = sum(1 for c in first if c)
    numeric_like = sum(1 for c in first if re.fullmatch(r"[₹$€£¥]?[-+]?\d[\d,]*(\.\d+)?%?", c.replace(" ", "")))
    has_header = len(rows) > 1 and non_empty >= max(2, max_cols // 2) and numeric_like <= max(1, max_cols // 3)

    if has_header:
        header: List[str] = []
        seen: Dict[str, int] = {}
        for i, c in enumerate(first, start=1):
            base = c or f"Column_{i}"
            if base in seen:
                seen[base] += 1
                header.append(f"{base}_{seen[base]}")
            else:
                seen[base] = 0
                header.append(base)
        df = pd.DataFrame(rows[1:], columns=header)
    else:
        df = pd.DataFrame(rows, columns=[f"Column_{i + 1}" for i in range(max_cols)])

    df = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all").dropna(axis=1, how="all").fillna("")
    return df


def image_to_dataframe(image_bytes: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    gray = decode_and_prepare(image_bytes)

    # Fast and usually more accurate for screenshots.
    layout_rows = build_rows_from_words(gray)
    layout_df = finalize_dataframe(layout_rows)

    # Grid OCR can help scanned tables; use it only if it clearly gives more structure.
    grid_rows = rows_from_grid(gray)
    grid_df = finalize_dataframe(grid_rows) if grid_rows else pd.DataFrame()

    if not grid_df.empty and len(grid_df.columns) >= len(layout_df.columns) and len(grid_df) >= max(2, len(layout_df) - 1):
        return grid_df, {"mode": "grid_cells", "rows": len(grid_df), "columns": len(grid_df.columns)}

    return layout_df, {"mode": "layout_words", "rows": len(layout_df), "columns": len(layout_df.columns)}


def dataframe_to_excel(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
        ws = writer.sheets["Sheet1"]
        for idx, col in enumerate(df.columns, start=1):
            values = [str(col)] + [str(v) for v in df.iloc[:, idx - 1].tolist()]
            ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = min(max(len(v) for v in values) + 2, 50)
    buf.seek(0)
    return buf


@app.on_event("startup")
def startup() -> None:
    try:
        log.info("Tesseract ready: %s", pytesseract.get_tesseract_version())
    except Exception as exc:
        log.error("Tesseract not working: %s", exc)


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "service": "image-to-excel", "version": app.version}


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception:
        version = "unavailable"
    return {"status": "ok", "tesseract": version}


@app.post("/convert")
async def convert(file: UploadFile = File(...)) -> StreamingResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"Unsupported file type: {file.content_type}")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        df, metrics = image_to_dataframe(contents)
        excel = dataframe_to_excel(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except pytesseract.TesseractNotFoundError as exc:
        raise HTTPException(status_code=500, detail="Tesseract OCR is not installed on server") from exc
    except Exception as exc:
        log.exception("Conversion failed")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {exc.__class__.__name__}") from exc

    out_name = (os.path.splitext(file.filename or "converted")[0] or "converted") + ".xlsx"
    headers = {
        "Content-Disposition": f'attachment; filename="{out_name}"',
        "X-Extraction-Mode": str(metrics.get("mode", "unknown")),
        "X-Rows": str(metrics.get("rows", len(df))),
        "X-Columns": str(metrics.get("columns", len(df.columns))),
    }
    return StreamingResponse(
        excel,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
