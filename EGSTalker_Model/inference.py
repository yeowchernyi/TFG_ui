import sys
import argparse
import subprocess
import os
import shutil
import glob
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 需要确保路径指向真正包含TensorFlow的环境
TF_PYTHON = "/home/xsj/.conda/envs/egstalker_tf_py39/bin/python"

def run(cmd: list[str], cwd: Path):
    print(f"执行命令: {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd))
    if p.returncode != 0:
        print(f"❌ 命令失败! 返回码: {p.returncode}")
        raise RuntimeError(f"Subprocess failed with code {p.returncode}")
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=str, default="3", help="指定使用的 GPU 编号")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"🚀 已设置 CUDA_VISIBLE_DEVICES = {args.gpu}")

    model_path = os.path.abspath(args.model_path)
    source_path = os.path.abspath(args.source_path)
    wav_path = Path(args.wav).resolve()
    npy_path = wav_path.with_suffix('.npy')

    # 1. 提取特征
    extract_script = BASE_DIR / "EGSTalker" / "data_utils" / "deepspeech_features" / "extract_ds_features.py"
    if not npy_path.exists():
        print(f"🚀 正在提取特征...")
        run([TF_PYTHON, str(extract_script), "--input", str(wav_path)], cwd=extract_script.parent)
        # 检查是否生成在脚本同级
        local_npy = extract_script.parent / npy_path.name
        if local_npy.exists(): shutil.move(str(local_npy), str(npy_path))

    if not npy_path.exists():
        raise FileNotFoundError(f"未能生成特征文件: {npy_path}")

    # 2. 执行渲染
    print("🎬 开始渲染视频...")
    render_cmd = [
        sys.executable, "EGSTalker/render.py",
        "--configs", args.configs,
        "--model_path", model_path,
        "--source_path", source_path,
        "--batch", str(args.batch),
        "--custom_wav", str(wav_path),
        "--custom_aud", str(npy_path),
        "--iteration", "-1",
        "--skip_train", "--skip_test"
    ]
    try:
        run(render_cmd, cwd=BASE_DIR)
    except Exception as e:
        print(f"渲染过程出错: {e}")
        raise e  # 抛出异常，让app.py接收到非零返回码

    # 3. 深度搜索生成的视频
    # 渲染器可能生成 .mov 或 .mp4，扩大搜索范围
    print("🔍 正在搜索生成的视频文件...")
    search_patterns = [
        os.path.join(args.model_path, "**", "*renders.mov"),
        os.path.join(args.model_path, "**", "*.mp4"),
        os.path.join(BASE_DIR, "EGSTalker", "result-video", "*.mp4")
    ]

    found_files = []
    for pattern in search_patterns:
        found_files.extend(glob.glob(pattern, recursive=True))

    if not found_files:
        # 最后尝试在整个 model_path 下找最近修改的文件
        all_files = glob.glob(os.path.join(args.model_path, "**", "*.*"), recursive=True)
        video_files = [f for f in all_files if f.endswith(('.mov', '.mp4'))]
        if video_files:
            found_files = video_files

    if not found_files:
        raise FileNotFoundError(f"渲染结束，但在 {args.model_path} 下没找到任何视频文件。")

    latest_video = max(found_files, key=os.path.getctime)
    print(f"✅ 找到视频: {latest_video}，正在转码...")

    # 4. 转码输出
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", latest_video,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", args.out
    ]
    run(ffmpeg_cmd, cwd=BASE_DIR)
    print(f"✨ 推理成功！最终文件: {args.out}")

if __name__ == "__main__":
    main()
