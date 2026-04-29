from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import pytesseract
import cv2
import pandas as pd
import uuid
import os
import io

# 🔥 IMPORTANT: tesseract path (top me hi hona chahiye)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Image to Excel API is running"}

@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    try:
        # Unique image filename
        image_name = f"input_{uuid.uuid4()}.png"

        # Save uploaded image
        with open(image_name, "wb") as f:
            f.write(await file.read())

        # Read image
        img = cv2.imread(image_name)

        # Preprocessing (important)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # OCR extract
        text = pytesseract.image_to_string(gray)

        # Convert text → list
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # DataFrame
        df = pd.DataFrame(lines, columns=["Extracted Data"])

        # 🔥 Save to memory (NO file corruption)
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=result.xlsx"}
        )

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Cleanup temp image
        if os.path.exists(image_name):
            os.remove(image_name)
