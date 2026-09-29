FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ARG REPO_URL=https://github.com/Ritik7nehra/DistributedGPT-CUDA-Optimized-Distributed-LLM-Training-Engine.git
RUN git clone --depth 1 "$REPO_URL" .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install fastapi uvicorn pyyaml numpy pydantic
RUN python src/train.py --data huggingface_space/sample_data/sample.txt --config configs/tiny.yaml --max-steps 300 --out-dir checkpoints
ENV CHECKPOINT=/app/checkpoints/ckpt.pt
EXPOSE 7860
CMD ["uvicorn","inference.api:app","--host","0.0.0.0","--port","7860"]