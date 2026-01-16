import os
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForVision2Seq,
    AutoModelForCausalLM, 
    AutoProcessor,
    AutoTokenizer,
    AutoImageProcessor,
    LlavaProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 嘗試引入新版 VLM Class
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

def load_tokenizer_and_processor(model_path, model_type):
    """載入 Tokenizer 和 Processor，包含失敗救援機制"""
    processor = None
    tokenizer = None
    
    print(">>> [Loader] Loading Tokenizer/Processor...")
    
    try:
        if model_type == "llm":
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            except Exception as e:
                print(f">>> [Loader] Fast Tokenizer failed: {str(e).splitlines()[0]}")
                print(">>> [Loader] Downgrading to Slow Tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        else:
            # VLM 邏輯
            try:
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                print(">>> [Loader] AutoProcessor failed, attempting manual assembly...")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
                image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
                processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)

            if hasattr(processor, "tokenizer"):
                tokenizer = processor.tokenizer
            elif tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer, processor

    except Exception as e:
        raise RuntimeError(f"Failed to load Tokenizer/Processor: {e}")

def detect_image_token(model_type, tokenizer, processor):
    """偵測並註冊 Image Token，並強制同步 Processor"""
    image_token_str = "<image>"
    if model_type == "vlm":
        try:
            token_obj = None
            if processor and hasattr(processor, "image_token"):
                token_obj = processor.image_token
            elif tokenizer and hasattr(tokenizer, "image_token"):
                token_obj = tokenizer.image_token
            
            # 提取純字串
            if token_obj is not None:
                if hasattr(token_obj, "content"):
                    image_token_str = token_obj.content
                else:
                    image_token_str = str(token_obj)
        except:
            pass
        
        print(f">>> [Loader] Detected Image Token: {image_token_str}")

        # 1. 註冊到 Tokenizer
        if image_token_str not in tokenizer.get_vocab():
            print(f">>> [Loader] Warning: '{image_token_str}' not in vocab. Adding as special token.")
            tokenizer.add_tokens([image_token_str], special_tokens=True)
            
        # 2. 同步 Tokenizer 到 Processor
        if processor and hasattr(processor, "tokenizer"):
            processor.tokenizer = tokenizer
            
        # 3. [關鍵修復] 強制將 Processor 的 image_token 屬性設為純字串
        # 解決 processing_gemma3.py 中 text.count(self.image_token) 失敗的問題
        if processor and hasattr(processor, "image_token"):
            try:
                # 某些 processor 的 image_token 是 property，不能直接設
                # 我們嘗試設定，如果失敗就算了
                processor.image_token = image_token_str
                print(f">>> [Loader] Patched processor.image_token to string: '{image_token_str}'")
            except:
                pass

        # 4. 同步 image_token_id
        image_token_id = tokenizer.convert_tokens_to_ids(image_token_str)
        if processor and hasattr(processor, "image_token_id"):
            # 有些模型允許 image_token_id 為 None，但若不一致我們嘗試同步
            current_id = getattr(processor, "image_token_id", None)
            if current_id != image_token_id:
                try:
                    processor.image_token_id = image_token_id
                    print(f">>> [Loader] Syncing processor.image_token_id: {current_id} -> {image_token_id}")
                except: pass
                
    return image_token_str

def load_model(model_path, model_type, quantization, strategy, dtype):
    """載入模型本體並設定量化與 LoRA"""
    print(">>> [Loader] Loading Model...")
    
    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }
    
    try:
        import flash_attn
        load_kwargs["attn_implementation"] = "flash_attention_2"
    except: pass
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    native_quant_config = getattr(config, "quantization_config", None)
    
    if quantization == "4bit":
        if native_quant_config is not None:
            print(f">>> [Loader] Using native quantization config ({type(native_quant_config).__name__}).")
        else:
            print(">>> [Loader] Applying BitsAndBytesConfig (4-bit)...")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )

    model_classes = []
    if model_type == "llm":
        model_classes.append(AutoModelForCausalLM)
    else:
        if AutoModelForImageTextToText: model_classes.append(AutoModelForImageTextToText)
        model_classes.append(AutoModelForVision2Seq)

    model = None
    last_error = None
    
    current_verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()

    for cls in model_classes:
        try:
            print(f">>> [Loader] Trying {cls.__name__}...")
            model = cls.from_pretrained(model_path, config=config, **load_kwargs)
            print(f">>> [Loader] Loaded with {cls.__name__}")
            break
        except Exception as e:
            last_error = e
    
    transformers.logging.set_verbosity(current_verbosity)

    if model is None:
        raise RuntimeError(f"Model load failed: {last_error}")

    # 通用設定
    model.config.use_cache = False
    model.is_parallelizable = True
    model.model_parallel = True
    
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    is_quantized = (quantization == "4bit") or (hasattr(model, "is_quantized") and model.is_quantized)
    if is_quantized:
        model = prepare_model_for_kbit_training(model)

    if strategy == "lora":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        arch = str(model.config).lower()
        if any(x in arch for x in ["llama", "mistral", "gemma", "qwen"]):
            target_modules.extend(["gate_proj", "up_proj", "down_proj"])
        
        config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    return model
