FROM nvcr.io/nvidia/pytorch:24.09-py3

# 設定工作目錄
WORKDIR /app

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# 關鍵：強制 bitsandbytes 使用內建的 CUDA 庫，避免去系統亂找
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# 1. 系統依賴
RUN apt-get update && apt-get install -y \
    git \
    libaio-dev \
    ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Python 套件安裝
RUN pip install --upgrade pip

# 安裝最新版的核心套件
# 這樣我們就能原生支援新的 tokenizer.json 格式
RUN pip install \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    "peft>=0.10.0" \
    "bitsandbytes>=0.43.1" \
    "datasets>=2.19.0" \
    scipy \
    einops \
    sentencepiece \
    protobuf

# 3. 安裝 Flash Attention 2 
RUN pip install flash-attn --no-build-isolation

# 4. 複製程式碼
COPY train_worker.py /app/
COPY benchmark_runner.py /app/

# 建立目錄
RUN mkdir -p /app/saves /app/data

CMD ["/bin/bash"]

