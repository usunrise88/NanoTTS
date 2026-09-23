FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image ~1 GB instead of ~6 GB.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.9.1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /work
ENV PYTHONUNBUFFERED=1
