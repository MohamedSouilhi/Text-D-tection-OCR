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

# Create directories to save files
output_dir = "./ocr_output"
original_dir = os.path.join(output_dir, "originals")
annotated_dir = os.path.join(output_dir, "annotated")

os.makedirs(original_dir, exist_ok=True)
os.makedirs(annotated_dir, exist_ok=True)

# Set up MLflow
mlflow.set_tracking_uri("file://./mlruns")
mlflow.set_experiment("TextDetectionAPI")

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a GPU and CUDA installed

# Initialize FastAPI app
app = FastAPI(title="Text Detection API")

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de détection de texte ! Utilisez /detect-text-multiple/ pour uploader des images ou des PDFs."}

@app.post("/detect-text-multiple/")
async def detect_text_multiple(files: List[UploadFile] = File(...), threshold: float = 0.5, return_images: bool = False):
    start_time = time.time()
    results = []
    total_detections = 0
    total_confidence = 0
    all_detections = []  # For retraining

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

            # Save the original file
            original_path = os.path.join(original_dir, filename)
            with open(original_path, 'wb') as f:
                f.write(file_data)

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

                            # Save the annotated image for this page
                            annotated_filename = f"{filename}_page_{page_num + 1}.png"
                            annotated_path = os.path.join(annotated_dir, annotated_filename)
                            Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)).save(annotated_path)
                            print(f"Saved annotated image: {annotated_path}")

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

                # Store detections for retraining
                for detection in detections:
                    detection_entry = {
                        "filename": filename,
                        "original_path": original_path,
                        "annotated_path": annotated_path if 'annotated_path' in locals() else None,
                        "text": detection["text"],
                        "confidence": detection["confidence"],
                        "bounding_box": detection["bounding_box"],
                        "page_number": detection["page_number"]
                    }
                    all_detections.append(detection_entry)

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

                # Save the annotated image
                annotated_filename = filename
                annotated_path = os.path.join(annotated_dir, annotated_filename)
                Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(annotated_path)
                print(f"Saved annotated image: {annotated_path}")

                # Store the result
                result = {
                    "filename": filename,
                    "detections": detections,
                    "image": image
                }
                results.append(result)

                # Store detections for retraining
                for detection in detections:
                    detection_entry = {
                        "filename": filename,
                        "original_path": original_path,
                        "annotated_path": annotated_path,
                        "text": detection["text"],
                        "confidence": detection["confidence"],
                        "bounding_box": detection["bounding_box"],
                        "page_number": detection["page_number"]
                    }
                    all_detections.append(detection_entry)

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

        # Log annotated images as artifacts in MLflow
        if return_images:
            for result in results:
                if result.get("image_base64"):
                    image_data = base64.b64decode(result["image_base64"])
                    image_pil = Image.open(io.BytesIO(image_data))
                    image_path = f"annotated_{result['filename']}"
                    image_pil.save(image_path)
                    mlflow.log_artifact(image_path)

        # Save detections as JSON for retraining
        detections_json_path = os.path.join(output_dir, "detections.json")
        with open(detections_json_path, 'w') as f:
            json.dump(all_detections, f, indent=4)
        print(f"Saved detections for retraining: {detections_json_path}")
        mlflow.log_artifact(detections_json_path)

        return {"status": "success", "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)