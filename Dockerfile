FROM python:3.11-slim

# env update and upgrade
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pytorch
RUN pip install --no-cache-dir \
        torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cpu

# requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code and model pt file.
COPY app/ ./app/
COPY model/ ./model/

# location of model pt file inside the container.
ENV MODEL_PATH=/app/model/best.pt

# expose port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]