from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pytesseract
import cv2
import pandas as pd
import uuid
import os

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Image to Excel API is running"}

@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    try:
        # Unique filenames
        image_name = f"input_{uuid.uuid4()}.png"
        output_name = f"output_{uuid.uuid4()}.xlsx"

        # Save uploaded image
        with open(image_name, "wb") as f:
            f.write(await file.read())

        # Read image
        img = cv2.imread(image_name)

        # Convert to grayscale (better OCR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # OCR extract
        text = pytesseract.image_to_string(gray)

        # Convert text lines → Excel
        lines = [line.strip() for line in text.split("\n") if line.strip() != ""]

        df = pd.DataFrame(lines, columns=["Extracted Data"])
        df.to_excel(output_name, index=False)

        return FileResponse(output_name, filename="result.xlsx")

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Cleanup (optional)
        if os.path.exists(image_name):
            os.remove(image_name)
