import base64
import time
from typing import List
import cv2
import easyocr
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import mlflow
import os
import PyPDF2
from pdf2image import convert_from_bytes
import json
import psycopg2
from minio import Minio
from minio.error import S3Error

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a GPU and CUDA installed

# Initialize FastAPI app
app = FastAPI(title="Text Detection API")

# PostgreSQL configuration (from environment variables)
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ocr_db")

# MinIO configuration (from environment variables)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "miniopassword")
MINIO_BUCKET = "ocr-bucket"

# Set up MLflow
mlflow.set_tracking_uri("file://./mlruns")
mlflow.set_experiment("TextDetectionAPI")

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # Set to True if using HTTPS
)

# Create MinIO bucket if it doesn't exist
try:
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
except S3Error as e:
    print(f"Error creating MinIO bucket: {e}")

# Initialize PostgreSQL connection
def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DB
    )

# Create table for storing detections if it doesn't exist
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255),
            original_url VARCHAR(255),
            annotated_url VARCHAR(255),
            text TEXT,
            confidence FLOAT,
            bounding_box JSONB,
            page_number INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

# Initialize the database on startup
init_db()

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de détection de texte ! Utilisez /detect-text-multiple/ pour uploader des images ou des PDFs."}

@app.post("/detect-text-multiple/")
async def detect_text_multiple(files: List[UploadFile] = File(...), threshold: float = 0.5, return_images: bool = False):
    start_time = time.time()
    results = []
    total_detections = 0
    total_confidence = 0

    with mlflow.start_run():
        # Log parameters in MLflow
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("return_images", return_images)
        mlflow.log_param("languages", "en")
        mlflow.log_param("contrast_ths", 0.1)
        mlflow.log_param("adjust_contrast", 0.5)

        for file in files:
            # Read the file
            file_data = await file.read()
            filename = file.filename
            content_type = file.content_type

            print(f"\nProcessing file: {filename} (Type: {content_type})")

            # Upload the original file to MinIO
            original_url = f"originals/{filename}"
            minio_client.put_object(
                MINIO_BUCKET,
                original_url,
                io.BytesIO(file_data),
                length=len(file_data),
                content_type=content_type
            )
            print(f"Uploaded original file to MinIO: {original_url}")

            # Check if the file is a PDF
            if content_type == "application/pdf":
                # Extract text directly from the PDF (if possible)
                try:
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_data))
                    pdf_text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text:
                            pdf_text += text

                    print(f"Extracted text from PDF: {pdf_text[:100]}...")

                    # If text was extracted directly, use it
                    if pdf_text.strip():
                        print("Using directly extracted text from PDF.")
                        detections = [{"text": pdf_text, "confidence": 1.0, "bounding_box": None, "page_number": None}]
                        total_detections += 1
                        total_confidence += 1.0
                        image = None  # No image to process
                    else:
                        # If the PDF is scanned (no selectable text), convert to images
                        print("No selectable text found. Converting PDF to images for OCR...")
                        images = convert_from_bytes(file_data)
                        print(f"Converted PDF to {len(images)} image(s).")
                        detections = []
                        for page_num, image in enumerate(images):
                            # Convert PIL image to OpenCV format
                            image_np = np.array(image)
                            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                            # Perform text detection with EasyOCR
                            ocr_results = reader.readtext(image_cv, contrast_ths=0.1, adjust_contrast=0.5)

                            # Process detections
                            for (bbox, text, confidence) in ocr_results:
                                if confidence >= threshold:
                                    top_left = [int(bbox[0][0]), int(bbox[0][1])]
                                    bottom_right = [int(bbox[2][0]), int(bbox[2][1])]
                                    detections.append({
                                        "text": text,
                                        "confidence": confidence,
                                        "bounding_box": {
                                            "top_left": top_left,
                                            "bottom_right": bottom_right
                                        },
                                        "page_number": page_num + 1
                                    })
                                    total_detections += 1
                                    total_confidence += confidence

                            # Draw bounding boxes on the image
                            for detection in detections:
                                if detection["page_number"] == page_num + 1:
                                    top_left = detection["bounding_box"]["top_left"]
                                    bottom_right = detection["bounding_box"]["bottom_right"]
                                    cv2.rectangle(image_cv, top_left, bottom_right, (0, 255, 0), 2)

                            # Save the annotated image to MinIO
                            annotated_filename = f"{filename}_page_{page_num + 1}.png"
                            annotated_url = f"annotated/{annotated_filename}"
                            _, buffer = cv2.imencode(".png", image_cv)
                            minio_client.put_object(
                                MINIO_BUCKET,
                                annotated_url,
                                io.BytesIO(buffer.tobytes()),
                                length=len(buffer),
                                content_type="image/png"
                            )
                            print(f"Uploaded annotated image to MinIO: {annotated_url}")

                        # Use the last page image for the result
                        if images:
                            image = image_cv
                        else:
                            image = None
                except Exception as e:
                    print(f"Error processing PDF {filename}: {str(e)}")
                    detections = [{"text": f"Error processing PDF: {str(e)}", "confidence": 0.0, "bounding_box": None, "page_number": None}]
                    image = None

                # Store the result for the PDF
                result = {
                    "filename": filename,
                    "detections": detections,
                    "image": image
                }
                results.append(result)

                # Store detections in PostgreSQL
                conn = get_db_connection()
                cursor = conn.cursor()
                for detection in detections:
                    cursor.execute("""
                        INSERT INTO detections (filename, original_url, annotated_url, text, confidence, bounding_box, page_number)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        filename,
                        original_url,
                        annotated_url if 'annotated_url' in locals() else None,
                        detection["text"],
                        detection["confidence"],
                        json.dumps(detection["bounding_box"]) if detection["bounding_box"] else None,
                        detection["page_number"]
                    ))
                conn.commit()
                cursor.close()
                conn.close()

            else:
                # Process image files
                image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)

                # Perform text detection
                ocr_results = reader.readtext(image, contrast_ths=0.1, adjust_contrast=0.5)

                # Process detections
                detections = []
                for (bbox, text, confidence) in ocr_results:
                    if confidence >= threshold:
                        top_left = [int(bbox[0][0]), int(bbox[0][1])]
                        bottom_right = [int(bbox[2][0]), int(bbox[2][1])]
                        detections.append({
                            "text": text,
                            "confidence": confidence,
                            "bounding_box": {
                                "top_left": top_left,
                                "bottom_right": bottom_right
                            },
                            "page_number": None
                        })
                        total_detections += 1
                        total_confidence += confidence

                # Draw bounding boxes on the image
                for detection in detections:
                    top_left = detection["bounding_box"]["top_left"]
                    bottom_right = detection["bounding_box"]["bottom_right"]
                    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

                # Save the annotated image to MinIO
                annotated_url = f"annotated/{filename}"
                _, buffer = cv2.imencode(".png", image)
                minio_client.put_object(
                    MINIO_BUCKET,
                    annotated_url,
                    io.BytesIO(buffer.tobytes()),
                    length=len(buffer),
                    content_type="image/png"
                )
                print(f"Uploaded annotated image to MinIO: {annotated_url}")

                # Store the result
                result = {
                    "filename": filename,
                    "detections": detections,
                    "image": image
                }
                results.append(result)

                # Store detections in PostgreSQL
                conn = get_db_connection()
                cursor = conn.cursor()
                for detection in detections:
                    cursor.execute("""
                        INSERT INTO detections (filename, original_url, annotated_url, text, confidence, bounding_box, page_number)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        filename,
                        original_url,
                        annotated_url,
                        detection["text"],
                        detection["confidence"],
                        json.dumps(detection["bounding_box"]) if detection["bounding_box"] else None,
                        detection["page_number"]
                    ))
                conn.commit()
                cursor.close()
                conn.close()

        # If return_images is true, convert images to base64
        if return_images:
            for result in results:
                image = result["image"]
                if image is not None:
                    _, buffer = cv2.imencode(".png", image)
                    image_base64 = base64.b64encode(buffer).decode("utf-8")
                    result["image_base64"] = image_base64
                else:
                    result["image_base64"] = None
                del result["image"]

        # Log metrics in MLflow
        processing_time = time.time() - start_time
        average_confidence = total_confidence / total_detections if total_detections > 0 else 0
        mlflow.log_metric("processing_time", processing_time)
        mlflow.log_metric("total_detections", total_detections)
        mlflow.log_metric("average_confidence", average_confidence)
        mlflow.log_metric("num_files_processed", len(files))

        # Log annotated images as artifacts in MLflow (if return_images is true)
        if return_images:
            for result in results:
                if result.get("image_base64"):
                    image_data = base64.b64decode(result["image_base64"])
                    image_pil = Image.open(io.BytesIO(image_data))
                    image_path = f"annotated_{result['filename']}"
                    image_pil.save(image_path)
                    mlflow.log_artifact(image_path)

        return {"status": "success", "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)