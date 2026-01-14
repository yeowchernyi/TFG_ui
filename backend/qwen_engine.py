import os
import threading
import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM

# 路径锚定
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

def get_idle_gpu():
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True, text=True)
        gpu_stats = []
        for line in output.strip().split('\n'):
            if not line.strip(): continue
            idx, free_mem = map(int, line.split(','))
            if idx not in [6, 7]:
                gpu_stats.append((idx, free_mem))
        best_gpu, max_free = max(gpu_stats, key=lambda x: x[1])
        return str(best_gpu)
    except Exception:
        return "0"

class LocalQwen:
    def __init__(self, model_id: str = None):
        # 自动定位到项目根目录下的 Qwen 文件夹
        default_path = os.path.join(PROJECT_ROOT, "Qwen", "Qwen2.5-0.5B-Instruct")
        self.model_id = model_id or os.environ.get("QWEN_MODEL_ID", default_path)
        self._lock = threading.Lock()
        self._ready = False
        self._mock_mode = False # 是否进入模拟模式
        self.tokenizer = None
        self.model = None

    def _lazy_load(self):
        if self._ready or self._mock_mode:
            return
        with self._lock:
            if self._ready or self._mock_mode:
                return

            print(f"🔍 正在检查模型路径: {self.model_id}")
            if not os.path.exists(self.model_id):
                print(f"⚠️ 警告: 在 {self.model_id} 未找到模型！")
                print("💡 将进入 Mock 模式（模拟对话），不占用显存。")
                self._mock_mode = True
                return

            try:
                target_gpu = get_idle_gpu()
                os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, use_fast=True, local_files_only=True
                )
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    dtype=dtype, # 修复了之前的警告
                    device_map="auto",
                    local_files_only=True
                )
                self._ready = True
                print(f"✅ [Qwen] 模型已成功加载到 GPU:{target_gpu}")
            except Exception as e:
                print(f"❌ 加载模型失败: {e}")
                self._mock_mode = True

    def chat(self, user_text: str, system_prompt: str = "助手", max_new_tokens: int = 256):
        self._lazy_load()

        if self._mock_mode:
            return f"[模拟回复] 我收到了你的消息：'{user_text}'。但由于本地未检测到 Qwen 模型，这是自动生成的回复。"

        # --- 正常的推理逻辑 ---
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)

        gen_ids = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

qwen = LocalQwen()
