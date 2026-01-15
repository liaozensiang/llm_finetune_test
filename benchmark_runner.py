import subprocess
import itertools
import csv
import time
import os
import torch
import re

# ==========================================================
# 1. 模型清單 (請填入您硬碟中的實際路徑)
# 格式: "顯示名稱": {"path": "絕對路徑", "type": "llm" 或 "vlm"}
# ==========================================================
MODELS = {
    "LLaVA-v1.5-7B": {"path": "models/vlm/llava-1.5-7b-hf", "type": "vlm"},
    "GPT-OSS-20B": {"path": "models/llm/gpt-oss-20b", "type": "llm"},  
    # "Gemma-2B":      {"path": "google/gemma-2b",                 "type": "llm"},
}

# ==========================================================
# 2. 測試變數 (排列組合矩陣)
# ==========================================================
CONFIG_MATRIX = {
    "seq_len": [2048, 4096, 8192, 16384], # 挑戰更長文本
    "img_res": [336, 672, 1344],          # 圖片解析度
    "strategy": ["lora", "full"],         # 微調策略
    "quantization": ["none", "4bit"],     # 量化載入
}

REPORT_FILE = "saves/benchmark_report.csv"

def get_gpu_info():
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if count > 0 else "Unknown"
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if count > 0 else 0
    return f"{count}x {name} ({vram:.1f}GB)"

def parse_vram_usage(output_str):
    match = re.search(r"Total Max Memory: (\d+\.\d+) GB", output_str)
    if match:
        return f"{match.group(1)} GB"
    return "N/A"

def run_benchmark():
    hardware_info = get_gpu_info()
    print(f"=== 開始通用基準測試 (Universal Benchmark) ===")
    print(f"硬體環境: {hardware_info}")
    
    if not os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Type", "Seq_Len", "Img_Res", "Strategy", "Quant", "Result", "Max_VRAM", "Note"])

    keys = CONFIG_MATRIX.keys()
    values = CONFIG_MATRIX.values()
    combinations = list(itertools.product(*values))
    
    total_tests_estimate = len(MODELS) * len(combinations)
    current_idx = 0

    for model_name, model_info in MODELS.items():
        model_path = model_info["path"]
        model_type = model_info["type"]

        for combo in combinations:
            current_idx += 1
            param_dict = dict(zip(keys, combo))
            
            seq_len = param_dict["seq_len"]
            img_res = param_dict["img_res"]
            strategy = param_dict["strategy"]
            quant = param_dict["quantization"]

            # === 智慧排程過濾器 ===
            
            # 1. LLM 不需要跑不同的圖片解析度
            # 如果是 LLM，強制只跑 img_res 的第一個值 (通常是最小的)，並在報告中標記 N/A
            # 這樣可以避免對同一個 LLM 跑 3 次重複的測試
            if model_type == "llm" and img_res != CONFIG_MATRIX["img_res"][0]:
                continue # 跳過重複測試

            display_res = img_res if model_type == "vlm" else "N/A"

            # 2. Full Fine-tune + 4bit 是無效組合
            if strategy == "full" and quant == "4bit":
                print(f"\n[SKIP] 無效組合: {model_name} | {strategy} + {quant}")
                continue
            
            print(f"\n[TESTING] {model_name} ({model_type}) | L={seq_len} | Res={display_res} | {strategy} | {quant}")

            # 構建指令
            cmd = [
                "python", "train_worker.py",
                "--model_path", model_path,
                "--model_type", model_type,
                "--seq_len", str(seq_len),
                "--strategy", strategy,
                "--quantization", quant
            ]
            
            if model_type == "vlm":
                cmd.extend(["--img_res", str(img_res)])

            # 執行
            start_time = time.time()
            result_status = "UNKNOWN"
            note = ""
            max_vram = "N/A"

            try:
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1200 # 20分鐘超時 (長文本可能很久)
                )
                
                if process.returncode == 0:
                    result_status = "PASS"
                    max_vram = parse_vram_usage(process.stdout)
                elif process.returncode == 137 or "out of memory" in process.stdout or "out of memory" in process.stderr:
                    result_status = "OOM"
                else:
                    result_status = "ERROR"
                    error_lines = process.stderr.strip().split('\n')[-5:]
                    note = " | ".join(error_lines)

            except subprocess.TimeoutExpired:
                result_status = "TIMEOUT"
            except Exception as e:
                result_status = "CRASH"
                note = str(e)

            duration = time.time() - start_time
            print(f"   -> {result_status} (VRAM: {max_vram}, Time: {duration:.1f}s)")
            
            with open(REPORT_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([model_name, model_type, seq_len, display_res, strategy, quant, result_status, max_vram, note])

            torch.cuda.empty_cache()

    print(f"\n=== 全部測試完成 ===\n報告: {REPORT_FILE}")

if __name__ == "__main__":
    run_benchmark()

