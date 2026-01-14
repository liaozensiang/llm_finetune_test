# 目標
一套自動化工具測試現有硬體條件是否能運行LLM & VLM微調
嘗試不同的 訓練方法 ＋ 量化 ＋ 資料長度 ＋ 圖片解析度 的排列組合

# 前置條件
安裝docker & nvidia-container-toolkit

# 資料結構(示範)
```
./
├── Dockerfile
├── benchmark_runner.py
├── models
│   ├── llm
│   │   └── gpt-oss-20b
│   └── vlm
│       └── llava-1.5-7b-hf
└── train_worker.py

```
# 使用說明
```bash
# Build image
sudo docker build -t xlm-benchmark:v1 .

# clone vlm (example)
sudo apt install git-lfs -y
cd models/vlm
git clone https://huggingface.co/llava-hf/llava-1.5-7b-hf
cd ../../ # 回到專案根目錄

# clone llm (example)
sudo apt install git-lfs -y
cd models/llm
git clone https://huggingface.co/openai/gpt-oss-20b
cd ../../ # 回到專案根目錄

# docker run
sudo docker run --gpus all \ 
                -it \
                --rm \
                --shm-size=64g \
                -v ./models:/app/models \
                -v ./results:/app/saves \
                xlm-benchmark:v1

# 在運行之前可以編輯一下 benchmark_runner.py
# 像是在MODELS裡加入你下載的的LLM & VLM模型
# 或是調整CONFIG_MATRIX裡的測試選項
python benchmark_runner.py
```

最後會在專案的根目錄裡新出現的results資料夾中得到一個csv檔
查看是否會遇到CUDA OOM
