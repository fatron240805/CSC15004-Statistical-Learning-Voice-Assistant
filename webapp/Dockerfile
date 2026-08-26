FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
# CPU-only torch/torchaudio wheels — HF Spaces free tier has no GPU, and the
# default PyPI wheel pulls multi-GB NVIDIA CUDA packages nobody needs here.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchaudio==2.4.1 \
    && pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY checkpoints ./checkpoints

ENV LOAD_SPEAKER_MODEL=1
EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
