import os
import shutil
import json
import torch
from PIL import Image
from datasets import load_dataset

def generate_temp_dataset(model_type, seq_len, img_res, image_token_str):
    """生成暫存的 JSON 和 圖片"""
    TEMP_DATA_DIR = "/tmp/benchmark_data"
    if os.path.exists(TEMP_DATA_DIR):
        shutil.rmtree(TEMP_DATA_DIR)
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    
    json_data = []
    img_path = None
    
    if model_type == "vlm":
        img_path = os.path.join(TEMP_DATA_DIR, "bench_img.jpg")
        Image.new('RGB', (img_res, img_res), color='red').save(img_path)
    
    prompt_text = "測試 " * (seq_len // 2)
    
    for i in range(10):
        entry = {"id": str(i)}
        if model_type == "vlm":
            entry["image"] = img_path
            # 這裡只存純文字 Prompt，Image Token 留給 preprocess_function 動態處理
            entry["conversations"] = [
                {"from": "human", "value": prompt_text},
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
        
    return json_path

def get_preprocess_function(tokenizer, processor, model_type, seq_len):
    def preprocess_function(examples):
        sources = examples['conversations']
        human_text = sources[0]['value'] 
        assistant_text = sources[1]['value']
        
        if model_type == "vlm":
            image_path = examples['image']
            try:
                image = Image.open(image_path).convert("RGB")
            except:
                image = Image.new('RGB', (336, 336), 'black')
            
            # 1. 動態獲取 Image Token
            image_token = "<image>"
            if hasattr(processor, "image_token"):
                token_obj = processor.image_token
                # 確保取到的是純字串
                if hasattr(token_obj, "content"):
                    image_token = token_obj.content
                else:
                    image_token = str(token_obj)
            elif hasattr(tokenizer, "image_token"): # Fallback to tokenizer if processor doesn't have it
                token_obj = tokenizer.image_token
                if hasattr(token_obj, "content"):
                    image_token = token_obj.content
                else:
                    image_token = str(token_obj)

            # 2. 構建 Prompt
            text = ""
            used_template = False
            
            # 嘗試使用 chat_template
            if hasattr(processor, "apply_chat_template"):
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": human_text}]},
                    {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
                ]
                try:
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    used_template = True
                except Exception:
                    pass
            
            if not used_template and hasattr(tokenizer, "apply_chat_template"):
                 messages = [
                    {"role": "user", "content": f"{image_token}\n{human_text}"},
                    {"role": "assistant", "content": assistant_text}
                ]
                 try:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    used_template = True
                 except:
                    pass

            # Fallback: 手動拼接 (如果 Template 失敗)
            if not used_template:
                text = f"USER: {image_token}\n{human_text}\nASSISTANT: {assistant_text}"
            
            # 雙重保險邏輯修正：
            # 如果使用了 Template，我們假設 Template 已經正確處理了 Image Token，不再手動添加。
            # 只有在未使用 Template 且檢測不到 Token 時才添加。
            if not used_template and image_token not in text:
                text = f"{image_token}\n{text}"

            inputs = processor(
                text=text, 
                images=image, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=seq_len
            )
            return {
                "input_ids": inputs.input_ids[0],
                "attention_mask": inputs.attention_mask[0],
                "pixel_values": inputs.pixel_values[0],
                "labels": inputs.input_ids[0]
            }
        else:
            # LLM 處理
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "user", "content": human_text},
                    {"role": "assistant", "content": assistant_text}
                ]
                try:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                except:
                    text = f"Human: {human_text}\nAssistant: {assistant_text}"
            else:
                text = f"Human: {human_text}\nAssistant: {assistant_text}"

            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_len
            )
            return {
                "input_ids": inputs.input_ids[0],
                "attention_mask": inputs.attention_mask[0],
                "labels": inputs.input_ids[0]
            }
    return preprocess_function

def get_data_collator(tokenizer, model_type):
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

        if model_type == "vlm" and "pixel_values" in features[0]:
            batch_pixel_values = [torch.tensor(f["pixel_values"]) for f in features]
            batch_pixel_values = torch.stack(batch_pixel_values)
            batch_data["pixel_values"] = batch_pixel_values

        return batch_data
    return data_collator
