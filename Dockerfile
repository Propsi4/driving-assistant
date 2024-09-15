FROM python:3.12

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install libgl1 -y

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

# Install YOLO weights
RUN gdown --fuzzy https://drive.google.com/file/d/1q9M9w4r16Bp7T6wh-lHXJPnCcA1rSNF-/view?usp=drive_link -O src/weights/traffic_signs_detection.pt

# Expose the port, must be the same as in the src/config/.env file(if you changed it)
EXPOSE 8000

CMD ["python", "-m", "src.models.api"]