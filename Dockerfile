FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY OCR.py .
<<<<<<< HEAD
COPY ocr_model.py .
=======
>>>>>>> a48b572f4a8d2ba73e80ab60d04f821a323ed455

EXPOSE 8000

CMD ["uvicorn", "OCR:app", "--host", "0.0.0.0", "--port", "8000"]