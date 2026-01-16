import argparse
import sys
import torch
import traceback
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# 匯入自定義模組
import env_utils
import model_loader
import data_handler

# 0. 優先執行環境修復
env_utils.apply_fixes()

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
print(f"    規格: Len={args.seq_len}, Res={args.img_res if args.model_type == 'vlm' else 'N/A'}, {args.strategy}, {args.quantization}")

def main():
    # 1. 決定精度
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f">>> [Worker] Precision: BF16={use_bf16}, FP16={use_fp16}")

    try:
        # 2. 載入 Tokenizer/Processor
        tokenizer, processor = model_loader.load_tokenizer_and_processor(args.model_path, args.model_type)
        
        # 3. 偵測 Image Token
        image_token_str = model_loader.detect_image_token(args.model_type, tokenizer, processor)

        # 4. 生成資料
        json_path = data_handler.generate_temp_dataset(
            args.model_type, args.seq_len, args.img_res, image_token_str
        )
        
        # 5. 載入模型
        model = model_loader.load_model(
            args.model_path, args.model_type, args.quantization, args.strategy, dtype
        )
        
        # Resize Embeddings (如果有新增 token)
        # 這裡需要小心 Gemma 3 的 vocab_size 坑
        current_vocab_size = getattr(model.config, "vocab_size", None)
        if current_vocab_size is None and hasattr(model.config, "text_config"):
            current_vocab_size = getattr(model.config.text_config, "vocab_size", None)
        if current_vocab_size is None:
             try: current_vocab_size = model.get_input_embeddings().num_embeddings
             except: pass

        if current_vocab_size is not None and len(tokenizer) > current_vocab_size:
            print(f">>> [Worker] Resizing embeddings: {current_vocab_size} -> {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

        # 6. 準備 Dataset 與 Trainer
        raw_dataset = load_dataset("json", data_files=json_path, split="train")
        preprocess_fn = data_handler.get_preprocess_function(tokenizer, processor, args.model_type, args.seq_len)
        train_dataset = raw_dataset.map(preprocess_fn)
        
        collator = data_handler.get_data_collator(tokenizer, args.model_type)

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
            data_collator=collator,
            tokenizer=tokenizer
        )

        print(">>> [Worker] 開始 Trainer 循環...")
        trainer.train()
        print(">>> [Worker] 訓練成功完成 (PASS)")
        
        # 顯存統計
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
