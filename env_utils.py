import os
import sys
import ctypes
import subprocess
import importlib.util
import re

def fix_bnb_path():
    """修復 bitsandbytes 的 CUDA 路徑與版本問題"""
    print(">>> [Env] Trying to fix bitsandbytes environment...")
    try:
        # 1. 偵測系統 CUDA 版本
        system_cuda_major = 12 
        try:
            nvcc_out = subprocess.check_output(["nvcc", "-V"], text=True)
            match = re.search(r"release (\d+)\.(\d+)", nvcc_out)
            if match:
                system_cuda_major = int(match.group(1))
                print(f">>> [Env] System CUDA Version detected: {match.group(1)}.{match.group(2)}")
        except:
            pass

        # 2. 修復 libcudart.so 路徑
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
            
        if not cuda_path and os.path.exists("/usr/local"):
            for root, _, files in os.walk("/usr/local"):
                for f in files:
                    if f.startswith("libcudart.so"):
                        cuda_path = root
                        break
                if cuda_path: break

        if cuda_path:
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if cuda_path not in current_ld:
                os.environ["LD_LIBRARY_PATH"] = f"{current_ld}:{cuda_path}" if current_ld else cuda_path
            
            try:
                for f in os.listdir(cuda_path):
                    if f.startswith("libcudart.so"):
                        ctypes.cdll.LoadLibrary(os.path.join(cuda_path, f))
                        break
            except: pass
        
        # 3. 強制設定 BNB_CUDA_VERSION
        bnb_spec = importlib.util.find_spec("bitsandbytes")
        if bnb_spec and bnb_spec.origin:
            bnb_dir = os.path.dirname(bnb_spec.origin)
            libs = [f for f in os.listdir(bnb_dir) if f.startswith("libbitsandbytes_cuda") and f.endswith(".so")]
            versions = [int(f.replace("libbitsandbytes_cuda", "").replace(".so", "")) for f in libs if f.replace("libbitsandbytes_cuda", "").replace(".so", "").isdigit()]
            
            if versions:
                valid_versions = [v for v in versions if v < (system_cuda_major + 1) * 10]
                target_ver = max(valid_versions) if valid_versions else max(versions)
                os.environ["BNB_CUDA_VERSION"] = str(target_ver)
                print(f">>> [Env] Force-setting BNB_CUDA_VERSION={target_ver}")

    except Exception as e:
        print(f">>> [Env] BNB Fix failed: {e}")

def check_triton():
    """檢查並自動安裝 Triton"""
    try:
        import triton
    except ImportError:
        print(">>> [Env] Warning: Triton not found. Auto-installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "triton"], stdout=subprocess.DEVNULL)
        except: pass

def apply_fixes():
    fix_bnb_path()
    check_triton()
