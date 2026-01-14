import os
import time
import subprocess
import shutil

def generate_video(data):
    """
    模拟视频生成逻辑：接收来自前端的参数，并返回一个视频路径。
    """
    print("[backend.video_generator] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    if data['model_name'] == "EGSTalker":
        try:
            REPO = "/home/xsj/work/repos/TFG_ui"
            EG_ROOT = os.path.join(REPO, "EGSTalker_Model", "EGSTalker")

            tts_wav = data["ref_audio"]
            source_path = data["source_path"]          # 强烈建议传 /home/xsj/work/repos/TFG_ui/data/obama
            model_path  = data["model_param"]
            configs     = data["configs"]

            job_id = data.get("job_id", str(int(time.time())))
            os.makedirs(os.path.join(REPO, "static", "audio"), exist_ok=True)

            wav_16k = os.path.join(REPO, "static", "audio", f"reply_{job_id}_16k.wav")
            npy_tmp = os.path.join(REPO, "static", "audio", f"reply_{job_id}.npy")

            PB = os.path.join(
                REPO, "data_utils", "deepspeech_features", "models",
                "deepspeech-0_1_0-b90017e8.pb"
            )

            # (0) 简单校验，早死早爽
            if not os.path.exists(tts_wav):
                raise FileNotFoundError(f"tts_wav 不存在: {tts_wav}")
            if not os.path.isdir(source_path):
                raise FileNotFoundError(f"source_path 不存在或不是目录: {source_path}")
            if not os.path.exists(PB):
                raise FileNotFoundError(f"DeepSpeech pb 不存在: {PB}")

            # (1) ffmpeg 转 16k mono s16
            cmd_ffmpeg = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", tts_wav,
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                wav_16k
            ]
            r = subprocess.run(cmd_ffmpeg, check=True, cwd=REPO, text=True, capture_output=True)
            if r.stdout:
                print("[ffmpeg stdout]\n", r.stdout[-2000:])
            if r.stderr:
                print("[ffmpeg stderr]\n", r.stderr[-2000:])

            # (2) 提取 deepspeech 特征 -> npy_tmp
            cmd_ds = [
                "bash", "-lc",
                f"PYTHONPATH={REPO} conda run -n egstalker_tf_py39 "
                f"python {REPO}/tools/extract_deepspeech_aud.py "
                f"--wav {wav_16k} --out_npy {npy_tmp} --pb {PB}"
            ]
            r = subprocess.run(cmd_ds, check=True, cwd=REPO, text=True, capture_output=True)
            if r.stdout:
                print("[ds stdout]\n", r.stdout[-2000:])
            if r.stderr:
                print("[ds stderr]\n", r.stderr[-2000:])

            # (3) custom_aud 放到 source_path 根目录
            npy_name = f"reply_{job_id}.npy"
            npy_for_render = os.path.join(source_path, npy_name)
            shutil.copy(npy_tmp, npy_for_render)
            print("[OK] wrote custom_aud:", npy_for_render)

            # (4) render
            cmd_render = [
                "python", "render.py",
                "--configs", configs,
                "--model_path", model_path,
                "--source_path", source_path,
                "--batch", str(data.get("batch", 16)),
                "--iteration", str(data.get("iteration", -1)),
                "--skip_train", "--skip_test",
                "--custom_aud", npy_name,
                "--custom_wav", wav_16k,
            ]
            r = subprocess.run(cmd_render, check=True, cwd=EG_ROOT, text=True, capture_output=True)
            if r.stdout:
                print("[render stdout]\n", r.stdout[-2000:])
            if r.stderr:
                print("[render stderr]\n", r.stderr[-2000:])

            # (5) 找输出 mp4（递归找最新 with_audio）
            search_root = os.path.join(model_path, "custom")
            latest_mp4, latest_t = None, -1
            for root, _, files in os.walk(search_root):
                for f in files:
                    if f.endswith(".mp4") and "with_audio" in f:
                        p = os.path.join(root, f)
                        t = os.path.getctime(p)
                        if t > latest_t:
                            latest_t, latest_mp4 = t, p

            if not latest_mp4:
                raise RuntimeError(f"没找到渲染输出 mp4, search_root={search_root}")

            os.makedirs(os.path.join(REPO, "static", "videos"), exist_ok=True)
            out_name = f"egstalker_{job_id}.mp4"
            destination_path = os.path.join("static", "videos", out_name)
            shutil.copy(latest_mp4, os.path.join(REPO, destination_path))
            print("[OK] video:", destination_path)
            return destination_path

        except subprocess.CalledProcessError as e:
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)
            print(f"[backend.video_generator] 命令执行失败: {e}")
            return os.path.join("static", "videos", "out.mp4")
        except Exception as e:
            print(f"[backend.video_generator] 其他错误: {e}")
            return os.path.join("static", "videos", "out.mp4")

    video_path = os.path.join("static", "videos", "out.mp4")
    print(f"[backend.video_generator] 视频生成完成，路径：{video_path}")
    return video_path
