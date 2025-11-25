FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Outils système de base + Tesseract + poppler
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev \
    ffmpeg \
    tesseract-ocr tesseract-ocr-lat tesseract-ocr-fra libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# PyTorch GPU (CUDA 12.1)
RUN pip3 install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Whisper
RUN pip3 install --no-cache-dir openai-whisper

# Librairies RAG / NLP / app
RUN pip3 install --no-cache-dir \
    "huggingface-hub<1.0.0" \
    "transformers<4.45.0" \
    "sentence-transformers==5.1.2" \
    chromadb \
    langchain \
    streamlit \
    pypdf \
    pytesseract \
    pillow \
    spacy \
    accelerate \
    requests

# Modèle spaCy FR (utile plus tard si besoin)
RUN python3 -m spacy download fr_core_news_md

WORKDIR /app

COPY rag_api.py /app/rag_api.py
COPY streamlit_gui.py /app/streamlit_gui.py

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_gui.py", "--server.port=8501", "--server.address=0.0.0.0"]
