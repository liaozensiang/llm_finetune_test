# LLM & VLM Hardware Benchmark Tool

## 🔰 Beginner's Guide (Start Here!)
If you are new to LLM/VLM fine-tuning or Docker, this section is for you.

### What does this tool do?

Training AI models requires a lot of Video RAM (VRAM) on your graphics card. This tool performs a "Stress Test" on your hardware. Instead of guessing, it actually tries to run training tasks with different settings (like Model Size, Image Resolution, and Text Length) to see if your GPU crashes (OOM - Out Of Memory) or succeeds.

### Key Concepts

* **LLM vs. VLM**: LLM is for text-only (like Llama 3). VLM is for images + text (like LLaVA).
* **OOM (Out Of Memory)**: This means your GPU VRAM was full, and the training failed. This is what we are testing for.
* **Quantization (4bit)**: A technique to make models smaller so they fit on smaller GPUs.
* **Docker**: We use Docker to create a "clean, isolated room" for the code to run in, so you don't have to worry about messing up your computer's system libraries.

### Quick Workflow
1. **Prepare**: Install Docker and the NVIDIA drivers.
2. **Download**: Get the AI models you want to test (using Git).
3. **Build**: Create the testing environment (Docker Image).
4. **Run**: Execute the benchmark script.
5. **Check**: Look at the generated CSV file to see green "PASS" or red "OOM".

## 🎯 Goal
An automated toolkit to test whether your current hardware specifications can handle LLM (Large Language Model) and VLM (Vision Language Model) fine-tuning.

It iterates through different combinations of:
* **Training Strategy** (LoRA vs. Full Fine-tune)
* **Quantization** (None vs. 4-bit)
* **Data Length** (Token sequence length)
* **Image Resolution** (For VLMs)

## 📋 Prerequisites

Ensure you have the following installed on your host machine:
* Docker Engine
* NVIDIA Container Toolkit (Essential for GPU access inside Docker)

## 📂 File Structure (Example)
```
./
├── Dockerfile              # The environment definition
├── benchmark_runner.py     # The main controller script
├── train_worker.py         # The actual training execution script
├── models                  # Folder to store your downloaded models
│   ├── llm
│   │   └── gpt-oss-20b     # Example LLM
│   └── vlm
│       └── llava-1.5-7b-hf # Example VLM
```

## 🚀 Usage Instructions

### 1. Build the Docker Image

This creates the isolated environment with all necessary libraries (PyTorch, Transformers, Flash Attention, etc.).
```bash
sudo docker build -t xlm-benchmark:v1 .
```

### 2. Download Models

You need to download the actual model weights from HuggingFace.
Note: Make sure you have `git-lfs` installed to handle large model files.

**Example: Cloning a VLM (LLaVA)**
```bash
# Install Git LFS if you haven't
sudo apt install git-lfs -y

# Navigate to the VLM directory
cd models/vlm
git clone [https://huggingface.co/llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)

# Return to project root
cd ../../ 
```

**Example: Cloning an LLM (GPT-OSS-20B)**
```bash
# Install Git LFS if you haven't
sudo apt install git-lfs -y

# Navigate to the LLM directory
cd models/llm
git clone [https://huggingface.co/openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

# Return to project root
cd ../../ 
```

### 3. Run the Benchmark

Launch the container and start the test.

**Run Command:**
```bash
# Run the container with GPU access
sudo docker run --gpus all \
                -it \
                --rm \
                --shm-size=64g \
                -v ./models:/app/models \
                -v ./results:/app/saves \
                xlm-benchmark:v1
```

**Configuration (Optional):**
Before running, you can edit `benchmark_runner.py` to:

1. Add the paths of the new models you downloaded into the MODELS dictionary.
2. Adjust the test combinations in CONFIG_MATRIX (e.g., test longer sequences).

```bash
# Inside the container, start the runner
python benchmark_runner.py
```

### 4. Analyze Results

After the process finishes, a new CSV file (e.g., `benchmark_report.csv`) will be generated in your local `./results` folder.

Open this file to view the matrix. Look for the Result column:
* **PASS**: Your hardware can handle this configuration.
* **OOM**: Your hardware ran out of memory.
* **ERROR**: Something went wrong.
