import os
import time
import requests
import subprocess
import shlex
import uuid
import shutil
import speech_recognition as sr

# 获取当前文件所在目录的上一级，即项目根目录 TFG_ui
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 定义子模块路径
EGS_MODEL_ROOT = os.path.join(BASE_DIR, "EGSTalker_Model")
EGS_CODE_ROOT = os.path.join(EGS_MODEL_ROOT, "EGSTalker")
DATA_OBAMA = os.path.join(BASE_DIR, "data", "obama")
OUT_OBAMA  = os.path.join(BASE_DIR, "output", "obama")
DEEPSPEECH_PB = os.path.join(BASE_DIR, "data_utils", "deepspeech_features", "models", "deepspeech-0_1_0-b90017e8.pb")
EXTRACT_SCRIPT = os.path.join(BASE_DIR, "tools", "extract_deepspeech_aud.py")

# LLM 接口配置
DEFAULT_LLM_API_URL = os.getenv("LLM_API_URL", "http://127.0.0.1:5001/api/chat/text")
ITERATION = "30000"

def _run_cmd(cmd, cwd=None):
    """封装的命令执行函数"""
    print(f"[CMD] 执行目录: {cwd or BASE_DIR}")
    print(f"[CMD] 命令内容: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
    if result.stderr:
        print(f"[CMD STDERR]: {result.stderr[-500:]}")
    return result

def qwen_text_to_obama_video(assistant_text: str) -> str:
    """核心逻辑：文字 -> 语音 -> 特征 -> 渲染视频"""
    # 1. 准备临时文件夹
    audio_dir = os.path.join(BASE_DIR, "static", "audio")
    video_dir = os.path.join(BASE_DIR, "static", "videos")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    job_id = uuid.uuid4().hex[:8]
    wav_tmp = os.path.join(audio_dir, f"reply_{job_id}.wav")
    wav_16k = os.path.join(audio_dir, f"reply_{job_id}_16k.wav")
    npy_out = os.path.join(audio_dir, f"reply_{job_id}.npy")

    try:
        # 1) TTS 生成语音
        print("正在生成 TTS 语音...")
        _run_cmd(["edge-tts", "--voice", "zh-CN-YunxiNeural", "--text", assistant_text, "--write-media", wav_tmp])

        # 2) FFmpeg 转码 (16k mono)
        _run_cmd(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_tmp, "-ar", "16000", "-ac", "1", wav_16k])

        # 3) 提取 DeepSpeech 特征 (使用 conda run 切换环境)
        print("正在提取音频特征...")
        extract_cmd = [
            "bash", "-lc",
            f"PYTHONPATH={BASE_DIR} conda run -n egstalker_tf_py39 "
            f"python {EXTRACT_SCRIPT} --wav {wav_16k} --out_npy {npy_out} --pb {DEEPSPEECH_PB}"
        ]
        _run_cmd(extract_cmd)

        # 4) 将特征文件复制到 data 目录供渲染器读取
        npy_name = f"reply_{job_id}.npy"
        shutil.copy(npy_out, os.path.join(DATA_OBAMA, npy_name))

        # 5) 运行渲染脚本
        print("正在进行视频渲染 (EGSTalker)...")
        render_cmd = [
            "python", "render.py",
            "--batch", "16",
            "--skip_train", "--skip_test",
            "--configs", "arguments/args.py",
            "--source_path", DATA_OBAMA,
            "--model_path", OUT_OBAMA,
            "--iteration", ITERATION,
            "--custom_wav", wav_16k,
            "--custom_aud", npy_name,
        ]
        _run_cmd(render_cmd, cwd=EGS_CODE_ROOT)

        # 6) 动态寻找生成的视频文件 (避开硬编码文件名)
        search_root = os.path.join(OUT_OBAMA, "custom")
        latest_mp4 = None
        latest_t = -1
        for root, _, files in os.walk(search_root):
            for f in files:
                if f.endswith(".mp4") and "with_audio" in f:
                    p = os.path.join(root, f)
                    t = os.path.getctime(p)
                    if t > latest_t:
                        latest_t, latest_mp4 = t, p

        if not latest_mp4:
            raise RuntimeError("渲染成功但未找到输出的 mp4 文件")

        # 7) 复制到静态资源目录
        final_video_name = f"chat_{job_id}.mp4"
        dst_path = os.path.join(video_dir, final_video_name)
        shutil.copy(latest_mp4, dst_path)
        
        # 返回给前端的相对路径
        return f"static/videos/{final_video_name}"

    except Exception as e:
        print(f"[ERROR] 视频生成失败: {e}")
        return "static/videos/out.mp4" # 返回一个保底视频

def chat_response(data):
    """主入口：处理前端请求"""
    print("[backend.chat_engine] 收到对话请求")
    
    # 语音转文字 (这里假设前端传来的音频在固定位置)
    input_audio = os.path.join(BASE_DIR, "SyncTalk", "audio", "aud.wav")
    input_text_path = os.path.join(BASE_DIR, "static", "text", "input.txt")
    
    # 1. ASR
    user_text = audio_to_text(input_audio, input_text_path)
    
    # 2. LLM
    output_text_path = os.path.join(BASE_DIR, "static", "text", "output.txt")
    assistant_text = get_ai_response_via_http(
        input_text_path=input_text_path,
        output_text_path=output_text_path
    )

    # 3. Video Generation
    video_url = qwen_text_to_obama_video(assistant_text)
    return video_url

def audio_to_text(input_audio, input_text_path):
    """
    从音频文件识别中文并写入文本文件。
    依赖 speech_recognition 的 Google 识别（需要联网）。
    """
    try:
        recognizer = sr.Recognizer()

        # 确保目录存在
        os.makedirs(os.path.dirname(input_text_path), exist_ok=True)

        with sr.AudioFile(input_audio) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)

            print("[backend.chat_engine] 正在识别语音...")

            text = recognizer.recognize_google(audio_data, language="zh-CN")

            with open(input_text_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"[backend.chat_engine] 语音识别完成，已保存到: {input_text_path}")
            print(f"[backend.chat_engine] 识别结果: {text}")

            return text

    except sr.UnknownValueError:
        print("[backend.chat_engine] 无法识别音频内容")
    except sr.RequestError as e:
        print(f"[backend.chat_engine] 语音识别服务错误: {e}")
    except FileNotFoundError:
        print(f"[backend.chat_engine] 音频文件不存在: {input_audio}")
    except Exception as e:
        print(f"[backend.chat_engine] 发生错误: {e}")

    # 失败时也写一个空串，避免下游读不到文件
    try:
        os.makedirs(os.path.dirname(input_text_path), exist_ok=True)
        with open(input_text_path, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass
    return ""


def get_ai_response_via_http(input_text_path, output_text_path, api_url=DEFAULT_LLM_API_URL, timeout_sec=180, system_prompt=""):
    """
    读取 input_text_path 的内容，POST 到本地大模型接口，写入 output_text_path。
    期望接口返回 JSON，其中包含 assistant_text。
    """
    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)

    with open(input_text_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        output = "（未识别到有效语音内容）"
        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"[backend.chat_engine] 输入为空，已写入默认回复到: {output_text_path}")
        return output

    try:
        payload = {
            "text": content,
            "system_prompt": system_prompt
        }
        resp = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()

        # 兼容你的接口结构：success / assistant_text
        if data.get("success") is False:
            raise RuntimeError(data.get("error") or "LLM API returned success=false")

        output = (data.get("assistant_text") or "").strip()
        if not output:
            output = "（模型未返回内容）"

    except Exception as e:
        output = f"（调用本地大模型失败：{e}）"
        print(f"[backend.chat_engine] {output}")

    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"[backend.chat_engine] 答复已保存到: {output_text_path}")
    return output
