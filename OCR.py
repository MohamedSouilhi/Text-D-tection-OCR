import os
import io
import json
import time
import base64
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import List
import mlflow
import psycopg2
from minio import Minio
from minio.error import S3Error
from ocr_model import detect_text_from_file
import traceback

# Initialize FastAPI app
app = FastAPI()

# Set up MLflow experiment
MLFLOW_EXPERIMENT_NAME = "OCR_Experiment"

# Create or get the experiment
try:
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
except Exception as e:
    print(f"Failed to initialize MLflow experiment: {str(e)}")
    raise e

# Environment variables
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ocr_db")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "miniopassword")
MINIO_BUCKET = "ocr-bucket"

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# Ensure MinIO bucket exists
try:
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
except S3Error as e:
    print(f"Failed to initialize MinIO bucket: {str(e)}")
    raise e

# Initialize PostgreSQL connection
try:
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DB
    )
    cursor = conn.cursor()

    # Create table for detections if it doesn't exist
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
            is_embedded_image BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
except Exception as e:
    print(f"Failed to connect to PostgreSQL: {str(e)}")
    raise e

# Custom exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def custom_exception_handler(request, exc):
    print(f"Unhandled error: {str(exc)}")
    print(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc)},
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Bienvenue sur l'API de détection de texte ! Utilisez /detect-text-multiple/ pour uploader des images ou des PDFs."
    }

# Endpoint to detect text from multiple files
@app.post("/detect-text-multiple/")
async def detect_text_multiple(
    files: List[UploadFile] = File(...),
    threshold: float = 0.5,
    return_images: bool = False
):
    start_time = time.time()
    results = []
    total_detections = 0
    total_confidence = 0

    # Start MLflow run with the specified experiment ID
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("num_files", len(files))
        mlflow.log_param("return_images", return_images)

        for file in files:
            filename = file.filename

            # Read file data
            file_data = await file.read()

            # Save original file to MinIO
            original_path = f"originals/{filename}"
            minio_client.put_object(MINIO_BUCKET, original_path, io.BytesIO(file_data), len(file_data))

            # Detect text using the EasyOCR model
            detections, file_detections, file_confidence, image_base64 = detect_text_from_file(file_data, filename, threshold, return_images)

            # Update totals
            total_detections += file_detections
            total_confidence += file_confidence

            # Save detections to PostgreSQL and MinIO
            for detection in detections:
                # Prepare data for PostgreSQL
                text = detection["text"]
                confidence = detection["confidence"]
                bounding_box = detection.get("bounding_box")
                page_number = detection.get("page_number")
                is_embedded_image = detection["is_embedded_image"]

                # Insert detection into PostgreSQL
                cursor.execute(
                    """
                    INSERT INTO detections (filename, original_url, text, confidence, bounding_box, page_number, is_embedded_image)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (filename, original_path, text, confidence, json.dumps(bounding_box) if bounding_box else None, page_number, is_embedded_image)
                )
                detection_id = cursor.fetchone()[0]
                conn.commit()

                # If there is an image to annotate (i.e., not a direct text extraction from PDF)
                if image_base64 and return_images:
                    # Decode base64 image to annotate
                    image_bytes = base64.b64decode(image_base64)
                    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

                    # Save annotated image to MinIO
                    if is_embedded_image:
                        annotated_filename = f"{filename}_page_{page_number}_embedded_{len([d for d in detections if d['page_number'] == page_number and d['is_embedded_image']])}.png"
                    elif page_number:
                        annotated_filename = f"{filename}_page_{page_number}.png"
                    else:
                        annotated_filename = filename
                    annotated_path = f"annotated/{annotated_filename}"

                    _, buffer = cv2.imencode(".png", image)
                    minio_client.put_object(MINIO_BUCKET, annotated_path, io.BytesIO(buffer.tobytes()), len(buffer))

                    # Update detection in PostgreSQL with annotated URL
                    cursor.execute(
                        """
                        UPDATE detections
                        SET annotated_url = %s
                        WHERE id = %s
                        """,
                        (annotated_path, detection_id)
                    )
                    conn.commit()

            # Prepare result for this file
            result = {
                "filename": filename,
                "detections": detections,
                "image_base64": image_base64 if return_images else None
            }
            results.append(result)

        # Log metrics to MLflow
        processing_time = time.time() - start_time
        average_confidence = total_confidence / total_detections if total_detections > 0 else 0
        mlflow.log_metric("processing_time", processing_time)
        mlflow.log_metric("total_detections", total_detections)
        mlflow.log_metric("average_confidence", average_confidence)

    # Return response
    return {
        "status": "success",
        "results": results
    }

# Cleanup on shutdown
@app.on_event("shutdown")
def shutdown_event():
    cursor.close()
    conn.close()