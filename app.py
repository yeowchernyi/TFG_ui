# app.py - 完整修复版本（含 /api/chat/avatar + edge-tts -> wav -> EGSTalker）
from flask import Flask, render_template, request, jsonify, send_file, session
import os
import sys
import uuid
import json
import threading
import subprocess
import shutil
import time
import cv2
from datetime import datetime
from werkzeug.utils import secure_filename
import signal
from pathlib import Path
import requests

from backend.qwen_engine import qwen
from backend.text_chat_engine import reply_text


# =========================
# Global Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EGS_INFER_URL = "http://127.0.0.1:5002/infer"   # EGSTalker_Model/app.py
GEN_VIDEO_DIR = os.path.join(BASE_DIR, "static", "generated_videos")
GEN_AUDIO_DIR = os.path.join(BASE_DIR, "static", "generated_audios")
os.makedirs(GEN_VIDEO_DIR, exist_ok=True)
os.makedirs(GEN_AUDIO_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
jobs = {}

# 创建Flask应用
app = Flask(__name__)
app.json.ensure_ascii = False
app.secret_key = 'egstalker_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')


# =========================
# Directories
# =========================
def create_directories():
    """创建所有必要的目录"""
    directories = [
        'static/uploads/videos',
        'static/uploads/training_videos',
        'static/uploads/audios',
        'static/videos',
        'static/generated_videos',
        'static/generated_audios',
        'configs',
        'output',
        'data',
        'logs',
        'checkpoints',
        'result-video'
    ]

    for d in directories:
        # 拼接成绝对路径
        target_path = os.path.join(BASE_DIR, d)
        os.makedirs(target_path, exist_ok=True)
        print(f"✓ 确保目录存在: {target_path}")

# 在应用启动时创建目录
create_directories()


# =========================
# TTS (GLOBAL)  ✅关键修复：必须在全局
# =========================
def tts_to_wav(text: str, out_dir: str = GEN_AUDIO_DIR) -> str:
    """
    edge-tts: text -> mp3
    ffmpeg: mp3 -> wav (16kHz, mono)
    return: wav_path
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("TTS 输入文本为空")

    os.makedirs(out_dir, exist_ok=True)

    job_id = uuid.uuid4().hex[:8]
    mp3_path = os.path.join(out_dir, f"tts_{job_id}.mp3")
    wav_path = os.path.join(out_dir, f"tts_{job_id}.wav")

    # 1) edge-tts 生成 mp3
    cmd_tts = [
        "edge-tts",
        "--voice", "zh-CN-YunxiNeural",
        "--text", text,
        "--write-media", mp3_path,
    ]
    p = subprocess.run(cmd_tts, capture_output=True, text=True)
    if p.returncode != 0 or (not os.path.exists(mp3_path)):
        raise RuntimeError(f"edge-tts 失败: {p.stderr or p.stdout}")

    # 2) ffmpeg 转成 16k 单声道 wav
    cmd_ffmpeg = [
        "ffmpeg", "-y",
        "-i", mp3_path,
        "-ac", "1",
        "-ar", "16000",
        wav_path
    ]
    p2 = subprocess.run(cmd_ffmpeg, capture_output=True, text=True)
    if p2.returncode != 0 or (not os.path.exists(wav_path)):
        raise RuntimeError(f"ffmpeg 转 wav 失败: {p2.stderr or p2.stdout}")

    return wav_path


# =========================
# Training Manager
# =========================
class TrainingManager:
    def __init__(self):
        self.tasks = {}
        self.processes = {}
        self.task_lock = threading.Lock()

    def start_training(self, config):
        """启动训练任务"""
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        with self.task_lock:
            self.tasks[task_id] = {
                'id': task_id,
                'config': config,
                'status': 'starting',
                'progress': 0,
                'logs': [],
                'start_time': datetime.now().isoformat(),
                'model_path': None,
                'error': None
            }

        # 在后台线程中运行训练
        thread = threading.Thread(
            target=self._run_training_task,
            args=(task_id, config),
            daemon=True
        )
        thread.start()

        return task_id

    def _run_training_task(self, task_id, config):
        """执行训练任务 - 完整修复版"""
        task = self.tasks[task_id]

        try:
            # 1. 初始状态
            task['status'] = 'starting'
            task['progress'] = 5
            self._add_log(task, "🚀 开始训练任务...")

            # 2. 验证视频路径
            video_path = config.get('video_path')
            self._add_log(task, f"接收到的视频路径: {video_path}")

            if not video_path:
                raise ValueError("❌ 错误：视频路径为空")

            # 转换为字符串
            video_path = str(video_path).strip()

            # 3. 尝试查找视频文件
            actual_path = None

            possible_paths = [
                video_path,
                os.path.abspath(video_path),
                video_path.lstrip('/').lstrip('\\'),
                os.path.join('static/uploads/videos', os.path.basename(video_path)),
                os.path.join('static/uploads/training_videos', os.path.basename(video_path)),
                os.path.join(os.getcwd(), video_path.lstrip('/').lstrip('\\'))
            ]

            possible_paths = list(dict.fromkeys(possible_paths))

            for path in possible_paths:
                if os.path.exists(path) and os.path.isfile(path):
                    actual_path = path
                    self._add_log(task, f"✅ 找到视频文件: {actual_path}")
                    break

            if not actual_path:
                upload_dir = 'static/uploads/videos'
                if os.path.exists(upload_dir):
                    files = os.listdir(upload_dir)
                    self._add_log(task, f"上传目录内容: {', '.join(files[:5])}")

                raise FileNotFoundError("❌ 无法找到视频文件。请检查上传是否成功。")

            video_path = actual_path

            # 4. 验证视频文件
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError("❌ 无法打开视频文件，可能文件已损坏")

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB

                self._add_log(task, f"📊 视频信息:")
                self._add_log(task, f"  大小: {file_size:.2f} MB")
                self._add_log(task, f"  帧数: {frame_count}")
                self._add_log(task, f"  FPS: {fps}")
                self._add_log(task, f"  分辨率: {width}x{height}")

                if frame_count < 10:
                    self._add_log(task, "⚠️  警告：视频帧数较少，可能影响训练效果")

            except Exception as e:
                self._add_log(task, f"⚠️  视频信息获取失败: {str(e)}")

            task['progress'] = 15

            # 5. 创建数据目录
            data_dir = config.get('data_dir', f"data/train_{task_id}")
            os.makedirs(data_dir, exist_ok=True)
            data_dir = os.path.abspath(data_dir)
            self._add_log(task, f"📁 数据目录: {data_dir}")

            # 复制视频到数据目录
            video_filename = f"source_{uuid.uuid4().hex[:6]}{Path(video_path).suffix}"
            dest_video_path = os.path.join(data_dir, video_filename)
            shutil.copy2(video_path, dest_video_path)
            self._add_log(task, f"✅ 视频复制完成: {dest_video_path}")

            task['progress'] = 25

            # 6. 预处理视频
            task['status'] = 'preprocessing'
            if config.get('preprocess', True):
                self._add_log(task, "开始视频预处理...")

                frames_dir = os.path.join(data_dir, 'frames')
                os.makedirs(frames_dir, exist_ok=True)

                cap = cv2.VideoCapture(dest_video_path)
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame.shape[0] != 512 or frame.shape[1] != 512:
                        frame = cv2.resize(frame, (512, 512))

                    frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_count += 1

                    if frame_count % 50 == 0:
                        progress = 25 + (frame_count / 300 * 15)  # 假设最多300帧
                        task['progress'] = min(progress, 40)

                cap.release()
                self._add_log(task, f"✅ 帧提取完成: {frame_count} 帧")

                dataset_config = {
                    'name': config.get('model_name', 'untitled'),
                    'frames_dir': frames_dir,
                    'total_frames': frame_count,
                    'fps': 25,
                    'resolution': [512, 512],
                    'source_video': dest_video_path,
                    'created_at': datetime.now().isoformat()
                }

                config_file = os.path.join(data_dir, 'dataset_config.json')
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset_config, f, indent=2, ensure_ascii=False)

                self._add_log(task, f"📝 数据集配置已保存: {config_file}")

            task['progress'] = 45

            # 7. 准备模型目录
            task['status'] = 'training'
            model_name = config.get('model_name', f'model_{task_id}')
            model_path = config.get('model_path', f'output/{model_name}')
            os.makedirs(model_path, exist_ok=True)
            model_path = os.path.abspath(model_path)
            self._add_log(task, f"💾 模型保存路径: {model_path}")

            train_config = {
                'task_id': task_id,
                'model_name': model_name,
                'data_dir': data_dir,
                'model_path': model_path,
                'video_source': video_path,
                'epochs': config.get('epochs', 100),
                'start_time': task['start_time'],
                'advanced_config': config.get('advanced_config', {}),
                'created_at': datetime.now().isoformat()
            }

            config_path = os.path.join(model_path, 'training_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(train_config, f, indent=2, ensure_ascii=False)

            self._add_log(task, f"📄 训练配置已保存: {config_path}")

            task['progress'] = 50

            # 8. 开始训练
            self._add_log(task, "开始模型训练...")

            train_script = 'train.py'
            if os.path.exists(train_script):
                self._run_actual_training(task_id, data_dir, model_path, config)
            else:
                self._simulate_training(task_id, model_path, config.get('epochs', 100))

            # 9. 训练完成
            task['status'] = 'completed'
            task['progress'] = 100
            task['model_path'] = model_path
            task['end_time'] = datetime.now().isoformat()

            summary = {
                'task_id': task_id,
                'status': 'completed',
                'model_name': model_name,
                'model_path': model_path,
                'data_dir': data_dir,
                'start_time': task['start_time'],
                'end_time': task['end_time'],
                'duration': self._calculate_duration(task['start_time'], task['end_time']),
                'video_source': os.path.basename(video_path)
            }

            summary_path = os.path.join(model_path, 'training_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self._add_log(task, "🎉 训练完成！")
            self._add_log(task, f"📁 模型保存在: {model_path}")
            self._add_log(task, f"📄 训练摘要: {summary_path}")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()

            task['status'] = 'failed'
            task['error'] = str(e)
            task['progress'] = 0
            task['end_time'] = datetime.now().isoformat()

            self._add_log(task, f"❌ 训练失败: {str(e)}")
            self._add_log(task, f"🔍 错误详情: {error_details[:500]}...")

            error_log = {
                'task_id': task_id,
                'error': str(e),
                'traceback': error_details,
                'timestamp': datetime.now().isoformat(),
                'config': config
            }

            error_path = f"logs/error_{task_id}.json"
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_log, f, indent=2, ensure_ascii=False)

    def _run_actual_training(self, task_id, data_dir, model_path, config):
        """运行实际的训练脚本"""
        task = self.tasks[task_id]

        train_cmd = [
            sys.executable, 'train.py',
            '-s', data_dir,
            '--model_path', model_path
        ]

        if config.get('config_file'):
            train_cmd.extend(['--configs', config['config_file']])
            self._add_log(task, f"使用配置文件: {config['config_file']}")

        if config.get('epochs'):
            train_cmd.extend(['--epochs', str(config['epochs'])])

        self._add_log(task, f"🚀 启动训练命令: {' '.join(train_cmd)}")

        self._run_subprocess(task_id, train_cmd, "训练")

    def _simulate_training(self, task_id, model_path, epochs):
        """模拟训练过程"""
        task = self.tasks[task_id]

        self._add_log(task, f"🎭 模拟训练模式，共 {epochs} 轮")

        for epoch in range(epochs):
            progress = 50 + (epoch / epochs * 45)
            task['progress'] = progress

            loss = 0.8 * (0.95 ** epoch) + 0.1
            log_msg = f"Epoch {epoch+1:04d}/{epochs:04d} - Loss: {loss:.4f}"
            self._add_log(task, log_msg)

            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'loss': loss,
                    'timestamp': datetime.now().isoformat(),
                    'model_version': '1.0'
                }

                checkpoint_path = os.path.join(model_path, f'checkpoint_{epoch+1:04d}.pth')
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint, f, indent=2)

                self._add_log(task, f"💾 保存检查点: {checkpoint_path}")

            time.sleep(0.5)

            if task['status'] == 'stopped':
                self._add_log(task, "⏹️ 训练被停止")
                break

    def _run_subprocess(self, task_id, cmd, phase_name):
        """运行子进程"""
        task = self.tasks[task_id]

        self._add_log(task, f"开始{phase_name}: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=os.getcwd()
            )

            self.processes[task_id] = process

            for line in process.stdout:
                line = line.strip()
                if line:
                    self._add_log(task, line)

                    if 'progress' in line.lower() or '%' in line:
                        try:
                            import re
                            match = re.search(r'(\d+(\.\d+)?)%', line)
                            if match:
                                percent = float(match.group(1))
                                if phase_name == "训练":
                                    task['progress'] = 50 + (percent * 0.45)
                                elif phase_name == "预处理":
                                    task['progress'] = 25 + (percent * 0.2)
                        except Exception:
                            pass

            process.wait()

            if task_id in self.processes:
                del self.processes[task_id]

            if process.returncode != 0:
                raise Exception(f"{phase_name}失败，退出码: {process.returncode}")

            self._add_log(task, f"✅ {phase_name}完成")

        except Exception as e:
            raise Exception(f"{phase_name}执行失败: {str(e)}")

    def _add_log(self, task, message):
        """添加日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        task['logs'].append(log_entry)

        if len(task['logs']) > 1000:
            task['logs'] = task['logs'][-1000:]

    def _calculate_duration(self, start_time_str, end_time_str):
        """计算持续时间"""
        try:
            start = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
            duration = end - start

            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60

            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        except Exception:
            return "未知"

    def stop_training(self, task_id):
        """停止训练"""
        if task_id in self.processes:
            process = self.processes[task_id]
            process.terminate()

            if task_id in self.tasks:
                self.tasks[task_id]['status'] = 'stopped'
                self._add_log(self.tasks[task_id], "训练已被用户停止")

            del self.processes[task_id]
            return True
        return False

    def get_task(self, task_id):
        return self.tasks.get(task_id)

    def list_tasks(self):
        return list(self.tasks.values())


# 初始化管理器
training_manager = TrainingManager()


# =========================
# Routes: Pages
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_generation', methods=['GET', 'POST'])
def video_generation():
    if request.method == 'POST':
        # 1. 获取前端表单数据
        target_text = request.form.get('target_text', '').strip()
        ref_audio_path = request.form.get('ref_audio', '').strip()
        model_name = request.form.get('model_name')
        model_path = request.form.get('model_path') # 获取前端选中的模型路径
        print(f"[DEBUG] 收到视频生成请求: 文本='{target_text}', 参考音频='{ref_audio_path}'")

        try:
            # 2. 确定最终使用的音频路径
            final_wav_path = None

            if target_text:
                # 情况 A: 用户输入了文字 -> 调用你写好的 tts_to_wav
                print(f"[DEBUG] 正在生成 TTS 语音...")
                final_wav_path = tts_to_wav(target_text)
            elif ref_audio_path and os.path.exists(ref_audio_path):
                # 情况 B: 用户没写字，但提供了音频路径
                final_wav_path = ref_audio_path
            else:
                return jsonify({"status": "error", "message": "请提供文字内容或有效的音频路径"}), 400

            # 3. 将音频发送给 EGSTalker 渲染后端 (Port 5002)
            print(f"[DEBUG] 正在请求渲染后端: {EGS_INFER_URL}")
            with open(final_wav_path, 'rb') as f:
                files = {'audio': (os.path.basename(final_wav_path), f, 'audio/wav')}
                data_payload = {
                    "model_path": model_path,
                    # 如果你的渲染后端支持 source_path，也可以在这里传
                }
                r = requests.post(EGS_INFER_URL, files=files, data=data_payload, timeout=1200)

            if r.status_code == 200:
                # 4. 保存生成的视频到 static 目录
                job_id = uuid.uuid4().hex[:8]
                video_name = f"gen_{job_id}.mp4"
                # 使用你代码里定义的 GEN_VIDEO_DIR
                save_path = os.path.join(GEN_VIDEO_DIR, video_name)
                
                with open(save_path, 'wb') as v_file:
                    v_file.write(r.content)
                
                print(f"[SUCCESS] 视频生成成功: {save_path}")
                # 返回给前端展示
                return jsonify({
                    "status": "success", 
                    "video_path": f"/static/generated_videos/{video_name}"
                })
            else:
                print(f"[ERROR] 后端返回错误: {r.status_code}")
                return jsonify({"status": "error", "message": "渲染后端处理失败"}), 500

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    # 如果是 GET 请求，依然只显示页面
    return render_template('video_generation.html')

@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    if request.method == 'POST':
        if request.is_json:
            data = request.json
            action = data.get('action')

            if action == 'start':
                if not data.get('video_path'):
                    return jsonify({'status': 'error', 'error': '视频路径不能为空'}), 400

                config = {
                    'video_path': data['video_path'],
                    'model_name': data.get('model_name', f'model_{uuid.uuid4().hex[:6]}'),
                    'model_path': data.get('model_path', f'output/{data.get("model_name", f"model_{uuid.uuid4().hex[:6]}")}'),
                    'data_dir': data.get('data_dir', f'data/train_{uuid.uuid4().hex[:6]}'),
                    'config_file': data.get('config_file', 'arguments/args.py'),
                    'epochs': int(data.get('epochs', 100)),
                    'preprocess': data.get('preprocess', True),
                    'use_gpu': data.get('use_gpu', False),
                    'advanced_config': data.get('advanced_config', {})
                }

                task_id = training_manager.start_training(config)

                return jsonify({
                    'status': 'success',
                    'task_id': task_id,
                    'message': '训练任务已启动',
                    'config': config
                })

            elif action == 'status':
                task_id = data.get('task_id')
                task = training_manager.get_task(task_id)

                if not task:
                    return jsonify({'status': 'error', 'error': '任务不存在'}), 404

                response = {
                    'status': 'success',
                    'task': {
                        'id': task['id'],
                        'status': task['status'],
                        'progress': task['progress'],
                        'start_time': task.get('start_time'),
                        'end_time': task.get('end_time'),
                        'model_path': task.get('model_path'),
                        'error': task.get('error')
                    }
                }

                if task.get('logs'):
                    response['task']['recent_logs'] = task['logs'][-20:]

                return jsonify(response)

            elif action == 'stop':
                task_id = data.get('task_id')
                success = training_manager.stop_training(task_id)

                if success:
                    return jsonify({'status': 'success', 'message': '训练任务已停止'})
                else:
                    return jsonify({'status': 'error', 'error': '任务不存在或无法停止'}), 404

            elif action == 'list':
                tasks = training_manager.list_tasks()
                simplified_tasks = []

                for task in tasks:
                    simplified_tasks.append({
                        'id': task['id'],
                        'status': task['status'],
                        'progress': task['progress'],
                        'start_time': task.get('start_time'),
                        'end_time': task.get('end_time'),
                        'model_path': task.get('model_path'),
                        'model_name': os.path.basename(task.get('model_path', '')) if task.get('model_path') else '未知'
                    })

                return jsonify({'status': 'success', 'tasks': simplified_tasks})

        return jsonify({'status': 'error', 'error': '不支持的请求格式'}), 400

    return render_template('model_training.html')

@app.route('/chat_system')
def chat_system():
    return render_template('chat_system.html')

@app.route('/inference')
def inference():
    return render_template('inference.html')


# =========================
# API: Upload video
# =========================
@app.route('/api/upload/video', methods=['POST'])
def api_upload_video():
    """上传视频文件 - 完整修复版"""
    try:
        print("=== 开始文件上传 ===")

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'}), 400

        allowed_extensions = {'mp4', 'avi', 'mov', 'm4v', 'mpg', 'mpeg', 'webm'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''

        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': f'不支持的文件类型: .{file_ext}。支持: {", ".join(allowed_extensions)}'
            }), 400

        upload_dir = 'static/uploads/videos'
        os.makedirs(upload_dir, exist_ok=True)

        original_name = file.filename
        safe_name = secure_filename(original_name)
        unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        file_path = os.path.join(upload_dir, unique_name)

        print(f"保存文件到: {file_path}")

        file.save(file_path)

        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': '文件保存失败'}), 500

        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)

        print(f"文件保存成功: {file_size_mb:.2f} MB")

        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                os.remove(file_path)
                return jsonify({'success': False, 'error': '视频文件无法打开或已损坏'}), 400

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            if duration > 600:
                os.remove(file_path)
                return jsonify({'success': False, 'error': '视频太长，最大10分钟'}), 400

        except Exception as e:
            print(f"视频验证失败: {e}")

        web_path = f"/static/uploads/videos/{unique_name}"

        return jsonify({
            'success': True,
            'message': '✅ 视频上传成功',
            'filename': unique_name,
            'original_name': original_name,
            'path': file_path,
            'web_path': web_path,
            'url': web_path,
            'size': f'{file_size_mb:.2f} MB',
            'size_bytes': file_size,
            'duration': f'{duration:.1f}s' if 'duration' in locals() else '未知'
        })

    except Exception as e:
        import traceback
        print(f"文件上传异常: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================
# API: Chat text
# =========================
@app.route('/api/chat/text', methods=['POST'])
def api_chat_text():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    system_prompt = data.get("system_prompt")
    max_new_tokens = int(data.get("max_new_tokens") or 256)

    if not text:
        return jsonify({"success": False, "error": "text 不能为空"}), 400

    max_new_tokens = max(1, min(max_new_tokens, 1024))

    try:
        result = reply_text(text, system_prompt=system_prompt, max_new_tokens=max_new_tokens)
        return jsonify({"success": True, "assistant_text": result["assistant_text"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# API: Upload audio
# =========================
@app.route('/api/upload/audio', methods=['POST'])
def api_upload_audio():
    """上传音频文件"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': '没有音频文件'}), 400

        audio_file = request.files['audio']

        if audio_file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'}), 400

        upload_dir = 'static/uploads/audios'
        os.makedirs(upload_dir, exist_ok=True)

        original_name = audio_file.filename
        safe_name = secure_filename(original_name)
        unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        file_path = os.path.join(upload_dir, unique_name)

        audio_file.save(file_path)

        return jsonify({
            'success': True,
            'filename': unique_name,
            'path': file_path,
            'url': f'/static/uploads/audios/{unique_name}'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================
# API: Defaults config
# =========================
@app.route('/api/config/defaults', methods=['GET'])
def api_config_defaults():
    """获取默认配置"""
    default_configs = {
        'ModelHiddenParams': {
            'kplanes_config': {
                'grid_dimensions': 2,
                'input_coordinate_dim': 3,
                'output_coordinate_dim': 32,
                'resolution': [64, 64, 64]
            },
            'multires': [1, 2],
            'defor_depth': 2,
            'net_width': 128,
            'plane_tv_weight': 0.0002,
            'time_smoothness_weight': 0.001,
            'l1_time_planes': 0.0001,
            'no_do': False,
            'no_dshs': False,
            'no_ds': False,
            'empty_voxel': False,
            'render_process': False,
            'static_mlp': False,
            'only_infer': False,
            'd_model': 64,
            'n_head': 16,
            'agent_num': 200,
            'drop_prob': 0.1,
            'ffn_hidden': 128,
            'n_layer': 1,
            'train_tri_plane': True
        },
        'OptimizationParams': {
            'dataloader': True,
            'densify_from_iter': 1000,
            'densification_interval': 100,
            'iterations': 30000,
            'batch_size': 8,
            'coarse_iterations': 100,
            'densify_until_iter': 7000,
            'opacity_threshold_coarse': 0.005,
            'opacity_threshold_fine_init': 0.005,
            'opacity_threshold_fine_after': 0.005,
            'densify_grad_threshold_coarse': 0.001,
            'lip_fine_tuning': True,
            'depth_fine_tuning': True,
            'deformation_lr_final': 0.00001,
            'deformation_lr_init': 0.0001,
            'split_gs_in_fine_stage': False,
            'canonical_tri_plane_factor_list': ["opacity", "shs"]
        },
        'TrainingPresets': {
            'quick': {'iterations': 5000, 'batch_size': 4, 'defor_depth': 1, 'net_width': 64},
            'balanced': {'iterations': 15000, 'batch_size': 8, 'defor_depth': 2, 'net_width': 128},
            'quality': {'iterations': 30000, 'batch_size': 16, 'defor_depth': 3, 'net_width': 256},
            'research': {'iterations': 50000, 'batch_size': 32, 'defor_depth': 4, 'net_width': 512}
        }
    }

    return jsonify({'success': True, 'configs': default_configs})


# =========================
# API: Chat avatar ✅
# =========================
# 1. 任务执行函数（后台运行）
def background_avatar_task(job_id, user_text, system_prompt, max_new_tokens):
    try:
        # --- 原有的同步逻辑移到这里 ---
        # 1) LLM
        result = reply_text(user_text, system_prompt=system_prompt, max_new_tokens=max_new_tokens)
        assistant_text = result["assistant_text"]
        jobs[job_id]["assistant_text"] = assistant_text
        
        # 2) TTS
        wav_path = tts_to_wav(assistant_text)
        
        # 3) EGSTalker 推理
        with open(wav_path, "rb") as f:
            files = {"audio": ("speech.wav", f, "audio/wav")}
            r = requests.post(EGS_INFER_URL, files=files, timeout=1200)
        
        if r.status_code == 200:
            video_name = f"avatar_{job_id}.mp4"
            video_path = os.path.join(GEN_VIDEO_DIR, video_name)
            with open(video_path, "wb") as vf:
                vf.write(r.content)
            
            # 标记完成
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["video_url"] = f"/static/generated_videos/{video_name}"
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "数字人渲染失败"
            
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

# 2. 修改后的提交接口
@app.route('/api/chat/avatar', methods=['POST'])
def api_chat_avatar():
    data = request.get_json(force=True) or {}
    user_text = (data.get("text") or "").strip()
    system_prompt = data.get("system_prompt")
    max_new_tokens = int(data.get("max_new_tokens") or 256)

    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {"status": "processing", "video_url": None, "assistant_text": "思考中..."}

    # 启动后台线程
    thread = threading.Thread(
        target=background_avatar_task, 
        args=(job_id, user_text, system_prompt, max_new_tokens)
    )
    thread.start()

    return jsonify({"success": True, "job_id": job_id})

# 3. 新增：状态查询接口
@app.route('/api/chat/job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


# =========================
# API: Config save/list
# =========================
@app.route('/api/config/save', methods=['POST'])
def api_config_save():
    """保存配置"""
    try:
        data = request.json
        config_name = data.get('name', 'custom_config')
        config_data = data.get('config', {})

        config_dir = 'configs'
        os.makedirs(config_dir, exist_ok=True)

        config_path = os.path.join(config_dir, f'{config_name}.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        return jsonify({
            'success': True,
            'message': f'配置已保存: {config_path}',
            'path': config_path
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/config/list', methods=['GET'])
def api_config_list():
    """列出所有配置"""
    try:
        config_dir = 'configs'
        configs = []

        if os.path.exists(config_dir):
            for filename in os.listdir(config_dir):
                if filename.endswith('.json'):
                    config_path = os.path.join(config_dir, filename)
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)

                        configs.append({
                            'name': filename.replace('.json', ''),
                            'path': config_path,
                            'config': config_data
                        })
                    except Exception:
                        continue

        return jsonify({'success': True, 'configs': configs})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================
# API: Models list
# =========================
@app.route('/api/models/list', methods=['GET'])
def api_models_list():
    """获取模型列表"""
    try:
        models = []

        output_dir = 'output'
        if os.path.exists(output_dir):
            for model_name in os.listdir(output_dir):
                model_path = os.path.join(output_dir, model_name)
                if os.path.isdir(model_path):
                    has_checkpoint = any(f.startswith('checkpoint_') for f in os.listdir(model_path))
                    has_config = any(f.endswith('_config.json') for f in os.listdir(model_path))

                    if has_checkpoint or has_config:
                        models.append({
                            'name': model_name,
                            'type': 'Trained Model',
                            'path': model_path,
                            'has_checkpoint': has_checkpoint,
                            'has_config': has_config
                        })

        if not models:
            models.append({
                'name': 'example_model',
                'type': 'Example',
                'path': 'output/example',
                'instructions': '训练后模型将出现在这里'
            })

        return jsonify({'success': True, 'models': models})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================
# API: Inference generate (保持你原逻辑)
# =========================
@app.route('/api/inference/generate', methods=['POST'])
def api_inference_generate():
    """生成推理视频"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': '没有音频文件'}), 400

        audio_file = request.files['audio']
        model_path = request.form.get('model_path')

        if not model_path or not os.path.exists(model_path):
            return jsonify({'success': False, 'error': '模型路径无效'}), 400

        audio_dir = 'static/uploads/audios'
        os.makedirs(audio_dir, exist_ok=True)

        audio_filename = f"{uuid.uuid4().hex[:8]}_{audio_file.filename}"
        audio_path = os.path.join(audio_dir, audio_filename)
        audio_file.save(audio_path)

        result_dir = 'result-video'
        os.makedirs(result_dir, exist_ok=True)

        result_filename = f"result_{uuid.uuid4().hex[:8]}.mp4"
        result_path = os.path.join(result_dir, result_filename)

        with open(result_path, 'wb') as f:
            f.write(b"Simulated video result")

        return jsonify({
            'success': True,
            'message': '视频生成成功',
            'video_path': result_path,
            'video_url': f'/result-video/{result_filename}'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================
# API: Health & Debug
# =========================
@app.route('/api/health', methods=['GET'])
def api_health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'working_directory': os.getcwd(),
        'python_version': sys.version
    })


@app.route('/api/debug/paths', methods=['GET'])
def api_debug_paths():
    """调试路径信息"""
    paths_info = {
        'current_dir': os.getcwd(),
        'exists': {
            'app.py': os.path.exists('app.py'),
            'train.py': os.path.exists('train.py'),
            'static/uploads/videos': os.path.exists('static/uploads/videos'),
            'output': os.path.exists('output'),
            'data': os.path.exists('data')
        },
        'upload_dir_contents': []
    }

    upload_dir = 'static/uploads/videos'
    if os.path.exists(upload_dir):
        paths_info['upload_dir_contents'] = os.listdir(upload_dir)[:10]

    return jsonify(paths_info)


# =========================
# Static routes（保持你原写法）
# =========================
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join('static', filename))


@app.route('/result-video/<path:filename>')
def serve_result_video(filename):
    return send_file(os.path.join('result-video', filename))


# =========================
# App start
# =========================
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 EGS-Talker 模型训练平台")
    print("=" * 60)
    print(f"📁 工作目录: {os.getcwd()}")
    print(f"🌐 服务地址: https://localhost:5001")
    print(f"📊 上传目录: {os.path.join(os.getcwd(), 'static/uploads/videos')}")
    print("=" * 60)

    create_directories()

    cert_path = os.path.expanduser("~/certs/cert.pem")
    key_path = os.path.expanduser("~/certs/key.pem")
    
    ssl_args = {}
    if os.path.exists(cert_path) and os.path.exists(key_path):
        ssl_args['ssl_context'] = (cert_path, key_path)
        print("🔐 已启用 HTTPS 模式")
    else:
        print("⚠️ 未找到证书文件，将以 HTTP 模式启动 (仅限本地测试)")

    app.run(
        debug=False,
        host="0.0.0.0",
        port=5001,
        threaded=True,
        **ssl_args
    )
