import cv2
import easyocr
import numpy as np
from PIL import Image
import io
import PyPDF2
from pdf2image import convert_from_bytes
import fitz
import base64

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

def detect_text_from_file(file_data, filename, threshold=0.5, return_images=False):
    """
    Detect text from a file (image or PDF) and return detections with optional annotated images.
    
    Args:
        file_data (bytes): File content in bytes.
        filename (str): Name of the file.
        threshold (float): Confidence threshold for OCR detections.
        return_images (bool): Whether to return base64-encoded annotated images.
    
    Returns:
        tuple: (detections, total_detections, total_confidence, image_base64)
    """
    content_type = "application/pdf" if filename.lower().endswith('.pdf') else "image"
    print(f"\nProcessing file: {filename} (Type: {content_type})")

    detections = []
    total_detections = 0
    total_confidence = 0
    image_base64 = None

    # Process PDF or image
    if content_type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_data))
            pdf_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    pdf_text += text

            print(f"Extracted text from PDF: {pdf_text[:100]}...")

            if pdf_text.strip():
                print("Using directly extracted text from PDF.")
                detections = [{"text": pdf_text, "confidence": 1.0, "bounding_box": None, "page_number": None, "is_embedded_image": False}]
                total_detections += 1
                total_confidence += 1.0
            else:
                print("No selectable text found. Converting PDF to images for OCR...")
                images = convert_from_bytes(file_data)
                print(f"Converted PDF to {len(images)} image(s).")

                print("Extracting embedded images from PDF...")
                pdf_document = fitz.open(stream=file_data, filetype="pdf")
                embedded_images = []
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        embedded_images.append((image, page_num + 1))
                        print(f"Extracted embedded image {img_index + 1} from page {page_num + 1}")

                images_to_process = [(img, page_num, False) for page_num, img in enumerate(images, 1)] + \
                                   [(img, page_num, True) for img, page_num in embedded_images]

                for image, page_num, is_embedded in images_to_process:
                    image_np = np.array(image)
                    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                    ocr_results = reader.readtext(image_cv, contrast_ths=0.1, adjust_contrast=0.5)

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
                                "page_number": page_num,
                                "is_embedded_image": is_embedded
                            })
                            total_detections += 1
                            total_confidence += confidence

                    if return_images:
                        for detection in detections:
                            if detection["page_number"] == page_num and detection["is_embedded_image"] == is_embedded:
                                top_left = detection["bounding_box"]["top_left"]
                                bottom_right = detection["bounding_box"]["bottom_right"]
                                cv2.rectangle(image_cv, top_left, bottom_right, (0, 255, 0), 2)

                        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
                        buffered = io.BytesIO()
                        image_pil.save(buffered, format="PNG")
                        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                pdf_document.close()
        except Exception as e:
            print(f"Error processing PDF {filename}: {str(e)}")
            detections = [{"text": f"Error processing PDF: {str(e)}", "confidence": 0.0, "bounding_box": None, "page_number": None, "is_embedded_image": False}]
    else:
        # Process image files
        image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        ocr_results = reader.readtext(image, contrast_ths=0.1, adjust_contrast=0.5)

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
                    "page_number": None,
                    "is_embedded_image": False
                })
                total_detections += 1
                total_confidence += confidence

        if return_images:
            for detection in detections:
                top_left = detection["bounding_box"]["top_left"]
                bottom_right = detection["bounding_box"]["bottom_right"]
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            buffered = io.BytesIO()
            image_pil.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return detections, total_detections, total_confidence, image_base64