import os
import sys
import ctypes
import importlib.util
import subprocess
import argparse
import torch
import json
import traceback
import shutil
from PIL import Image

# === 環境檢查 (保留最基本的 BNB 路徑修復，因為這是 Linux 環境配置問題) ===
def _fix_bnb_path():
    try:
        candidates = [
            "/usr/local/cuda/lib64", "/usr/local/cuda/lib",
            "/usr/lib/x86_64-linux-gnu", "/usr/lib/wsl/lib",
            "/usr/local/lib", "/usr/lib",
        ]
        cuda_path = None
        for p in candidates:
            if not os.path.exists(p): continue
            for f in os.listdir(p):
                if f.startswith("libcudart.so"):
                    cuda_path = p
                    break
            if cuda_path: break
            
        if cuda_path:
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if cuda_path not in current_ld:
                os.environ["LD_LIBRARY_PATH"] = f"{current_ld}:{cuda_path}" if current_ld else cuda_path
            try:
                # 嘗試預先載入，幫助 bitsandbytes 找到庫
                ctypes.cdll.LoadLibrary(os.path.join(cuda_path, "libcudart.so"))
            except: pass
    except: pass

def _check_triton():
    # 僅檢查，不 Patch。新版 Docker 應該自帶 Triton。
    try:
        import triton
    except ImportError:
        print(">>> [Worker] Warning: Triton not found. Auto-installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "triton"], stdout=subprocess.DEVNULL)
        except: pass

_fix_bnb_path()
_check_triton()

# === 正常 Import (不再需要 Monkey Patch) ===
import transformers
from transformers import (
    AutoConfig,
    AutoModelForVision2Seq,
    AutoModelForCausalLM, 
    AutoProcessor,
    AutoTokenizer,
    AutoImageProcessor,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# === 參數解析 ===
parser = argparse.ArgumentParser(description="Universal Benchmark Worker")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--model_type", type=str, choices=["llm", "vlm"], required=True)
parser.add_argument("--seq_len", type=int, default=1024)
parser.add_argument("--img_res", type=int, default=336)
parser.add_argument("--num_images", type=int, default=1)
parser.add_argument("--strategy", type=str, choices=["lora", "full"], default="lora")
parser.add_argument("--quantization", type=str, choices=["none", "4bit"], default="none")
parser.add_argument("--output_dir", type=str, default="saves/benchmark_temp")
args = parser.parse_args()

print(f">>> [Worker] 啟動測試: {args.model_path} [{args.model_type.upper()}]")
print(f"    環境版本: PyTorch {torch.__version__}, Transformers {transformers.__version__}")
print(f"    規格: Len={args.seq_len}, Res={args.img_res if args.model_type == 'vlm' else 'N/A'}, {args.strategy}, {args.quantization}")

def main():
    # 0. 精度設定
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f">>> [Worker] Precision: BF16={use_bf16}, FP16={use_fp16}")

    processor = None
    tokenizer = None
    
    # 1. 準備 Tokenizer / Processor
    try:
        print(">>> [Worker] 載入 Tokenizer/Processor...")
        if args.model_type == "llm":
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            except Exception:
                print(">>> [Worker] Fast Tokenizer 失敗，降級為 Slow...")
                tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
        else:
            # VLM
            try:
                processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
            except Exception:
                print(">>> [Worker] AutoProcessor 失敗，嘗試手動組裝...")
                tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
                image_processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)
                # 針對 LLaVA 的 fallback
                processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)

            if hasattr(processor, "tokenizer"):
                tokenizer = processor.tokenizer
            elif tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"!!! [Worker] 載入失敗: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. 生成暫存資料集
    TEMP_DATA_DIR = "/tmp/benchmark_data"
    if os.path.exists(TEMP_DATA_DIR):
        shutil.rmtree(TEMP_DATA_DIR)
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    
    json_data = []
    img_path = None
    if args.model_type == "vlm":
        img_path = os.path.join(TEMP_DATA_DIR, "bench_img.jpg")
        Image.new('RGB', (args.img_res, args.img_res), color='red').save(img_path)
    
    prompt_text = "測試 " * (args.seq_len // 2)
    for i in range(10):
        entry = {"id": str(i)}
        if args.model_type == "vlm":
            entry["image"] = img_path
            entry["conversations"] = [
                {"from": "human", "value": f"<image>\n{prompt_text}"},
                {"from": "gpt", "value": "收到"}
            ]
        else:
            entry["conversations"] = [
                {"from": "human", "value": prompt_text},
                {"from": "gpt", "value": "收到"}
            ]
        json_data.append(entry)
    
    json_path = os.path.join(TEMP_DATA_DIR, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False)

    # 3. 載入模型
    try:
        print(">>> [Worker] 載入模型...")
        
        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True
        }
        
        # 嘗試啟用 Flash Attention 2
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except: pass
        
        if args.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb_config

        if args.model_type == "llm":
            model_class = AutoModelForCausalLM
        else:
            model_class = AutoModelForVision2Seq

        # === 解決 transformers 序列化 Bug ===
        # 有些版本在 quantization_config 為 None 時 logging 會報錯
        # 我們先載入 Config，並在模型載入期間暫時隱藏 log
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        
        # 暫時提高 log level 避免 config 打印出錯
        current_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        
        model = model_class.from_pretrained(
            args.model_path,
            config=config, # 明確傳入 config
            **load_kwargs 
        )
        
        # 恢復 log level
        transformers.logging.set_verbosity(current_verbosity)
        
        # 通用設定
        model.config.use_cache = False
        model.is_parallelizable = True
        model.model_parallel = True
        
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        if args.quantization == "4bit":
            model = prepare_model_for_kbit_training(model)

        if args.strategy == "lora":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            arch = str(model.config).lower()
            if any(x in arch for x in ["llama", "mistral", "gemma", "qwen"]):
                target_modules.extend(["gate_proj", "up_proj", "down_proj"])
            
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)
            model.print_trainable_parameters()
            
    except Exception as e:
        print(f"!!! [Worker] 模型載入失敗: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. 資料前處理
    def preprocess_function(examples):
        sources = examples['conversations']
        human_text = sources[0]['value']
        
        if args.model_type == "vlm":
            image_path = examples['image']
            try:
                image = Image.open(image_path).convert("RGB")
            except:
                image = Image.new('RGB', (336, 336), 'black')
            text = f"USER: {human_text}\nASSISTANT: {sources[1]['value']}"
            inputs = processor(
                text=text, 
                images=image, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=args.seq_len
            )
            return {
                "input_ids": inputs.input_ids[0],
                "attention_mask": inputs.attention_mask[0],
                "pixel_values": inputs.pixel_values[0],
                "labels": inputs.input_ids[0]
            }
        else:
            text = f"Human: {human_text}\nAssistant: {sources[1]['value']}"
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=args.seq_len
            )
            return {
                "input_ids": inputs.input_ids[0],
                "attention_mask": inputs.attention_mask[0],
                "labels": inputs.input_ids[0]
            }

    raw_dataset = load_dataset("json", data_files=json_path, split="train")
    train_dataset = raw_dataset.map(preprocess_function)

    # 5. Data Collator
    def data_collator(features):
        batch_input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        batch_labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        batch_attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
        
        batch_data = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels
        }

        if args.model_type == "vlm" and "pixel_values" in features[0]:
            batch_pixel_values = [torch.tensor(f["pixel_values"]) for f in features]
            batch_pixel_values = torch.stack(batch_pixel_values)
            batch_data["pixel_values"] = batch_pixel_values

        return batch_data

    # 6. 訓練配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_steps=5,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    print(">>> [Worker] 開始 Trainer 循環...")
    try:
        trainer.train()
        print(">>> [Worker] 訓練成功完成 (PASS)")
        
        total_mem = 0
        gpu_details = []
        for i in range(torch.cuda.device_count()):
            mem_i = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
            total_mem += mem_i
            gpu_details.append(f"{mem_i:.2f}")
        
        print(f"GPU Details: {gpu_details}")
        print(f"Total Max Memory: {total_mem:.2f} GB")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(">>> [Worker] 捕捉到 OOM")
            sys.exit(137)
        else:
            print(f"!!! [Worker] 訓練 RuntimeError: {e}")
            traceback.print_exc() 
            sys.exit(1)
    except Exception as e:
        print(f"!!! [Worker] 訓練未知錯誤: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

