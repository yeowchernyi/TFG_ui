import os
import uuid
import subprocess
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import traceback
from pathlib import Path

app = Flask(__name__)
CORS(app)

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROJECT_ROOT = os.path.dirname(BASE_DIR)

# 修改后的 build_command，支持动态路径
def build_command(wav_path: str, out_mp4_path: str, model_path: str, gpu_id: str = "3") -> list[str]:
    # 1. 确定模型路径 (如果主控端没传，默认用 obama)
    if not model_path:
        model_path = os.path.join(PROJECT_ROOT, "output", "obama")

    # 2. 自动推断 source_path (原始视频数据)
    # 逻辑：如果模型在 output/obama，那么原始数据就在 data/obama
    model_name = os.path.basename(model_path.rstrip('/'))
    source_path = os.path.join(PROJECT_ROOT, "data", model_name)

    return [
        "python", "inference.py",
        "--wav", os.path.abspath(wav_path),
        "--configs", "EGSTalker/arguments/args.py",
        "--model_path", os.path.abspath(model_path),
        "--source_path", os.path.abspath(source_path),
        "--out", os.path.abspath(out_mp4_path),
        "--batch", "8",
        "--gpu", gpu_id
    ]

def get_idle_gpu():
    """
    自动获取最空闲的 GPU 编号
    """
    try:
        # 执行 nvidia-smi 查询每块卡的索引和剩余显存 (单位 MiB)
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True, text=True)

        # 解析输出：[(id, free_memory), ...]
        gpu_stats = []
        for line in output.strip().split('\n'):
            idx, free_mem = map(int, line.split(','))
            if idx not in [6, 7]:
                gpu_stats.append((idx, free_mem))

        # 按剩余显存从大到小排序，取第一名
        best_gpu, max_free = max(gpu_stats, key=lambda x: x[1])

        print(f"📊 GPU 状态监控: 最空闲显卡为 {best_gpu} 号 (剩余 {max_free} MiB)")
        return str(best_gpu)

    except Exception as e:
        print(f"⚠️ 无法获取 GPU 状态 ({e})，将回退到默认显卡 3")
        return "3"

@app.route("/infer", methods=["POST"])
def infer():
    print("\n" + "="*40)
    print("[DEBUG] 收到渲染请求")

    if "audio" not in request.files:
        return jsonify({"error": "missing form field: audio"}), 400

    # 获取主控端传过来的模型路径
    model_path = request.form.get("model_path")
    f = request.files["audio"]
    job_id = uuid.uuid4().hex[:8]
    wav_path = os.path.join(UPLOAD_DIR, f"{job_id}.wav")
    out_mp4_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    gpu_id = request.form.get("gpu", "3")
    f.save(wav_path)
    # 1. 动态获取 GPU 编号
    requested_gpu = request.form.get("gpu")
    if requested_gpu:
        gpu_id = requested_gpu
        print(f"[INFO] 用户手动指定使用 GPU: {gpu_id}")
    else:
        # 如果用户没传，自动找最空闲的
        gpu_id = get_idle_gpu()
        print(f"[INFO] 自动选择最空闲 GPU: {gpu_id}")

    # 2. 检查是否违规（双重保险）
    if gpu_id in ["6", "7"]:
        return jsonify({"error": "GPU 6 and 7 are reserved and cannot be used."}), 403
    # 将 model_path 传入命令构建函数
    cmd = build_command(wav_path, out_mp4_path, model_path, gpu_id)

    try:
        print(f"[DEBUG] 正在执行命令: {' '.join(cmd)}")
        p = subprocess.run(cmd, cwd=BASE_DIR, timeout=1200)

        if p.returncode != 0:
            print(f"[ERROR] inference.py 返回错误码: {p.returncode}")
            print(f"[STDERR]: {p.stderr}")
            return jsonify({"error": "inference failed", "details": p.stderr}), 500

        if os.path.exists(out_mp4_path) and os.path.getsize(out_mp4_path) > 0:
            print(f"[SUCCESS] 渲染成功: {out_mp4_path}")
            return send_file(out_mp4_path, mimetype="video/mp4")
        else:
            return jsonify({"error": "Output file not found"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True, threaded=True
