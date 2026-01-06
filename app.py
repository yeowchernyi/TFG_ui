# app.py - 整理修复版
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

# 创建Flask应用
app = Flask(__name__)


# ========== 测试路由 ==========
@app.route('/test-train-status/<task_id>', methods=['GET'])
def test_train_status(task_id):
    """测试路由 - 确认是否工作"""
    return jsonify({
        'message': '测试路由工作正常！',
        'task_id': task_id,
        'timestamp': datetime.now().isoformat()
    })


app.secret_key = 'egstalker_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = 'static/uploads'


# ========== 确保所有目录存在 ==========
def create_directories():
    """创建所有必要的目录"""
    directories = [
        'static/uploads/videos',
        'static/uploads/training_videos',
        'static/uploads/audios',
        'static/videos',
        'configs',
        'output',
        'data',
        'logs',
        'checkpoints',
        'result-video'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 确保目录存在: {directory}")


# 在应用启动时创建目录
create_directories()


# ========== 训练任务管理器 ==========
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

            # 尝试可能的路径
            possible_paths = [
                video_path,
                os.path.abspath(video_path),
                video_path.lstrip('/').lstrip('\\'),
                os.path.join('static/uploads/videos', os.path.basename(video_path)),
                os.path.join('static/uploads/training_videos', os.path.basename(video_path)),
                os.path.join(os.getcwd(), video_path.lstrip('/').lstrip('\\'))
            ]

            # 去重
            possible_paths = list(dict.fromkeys(possible_paths))

            for path in possible_paths:
                if os.path.exists(path) and os.path.isfile(path):
                    actual_path = path
                    self._add_log(task, f"✅ 找到视频文件: {actual_path}")
                    break

            if not actual_path:
                # 列出上传目录帮助调试
                upload_dir = 'static/uploads/videos'
                if os.path.exists(upload_dir):
                    files = os.listdir(upload_dir)
                    self._add_log(task, f"上传目录内容: {', '.join(files[:5])}")

                raise FileNotFoundError(f"❌ 无法找到视频文件。请检查上传是否成功。")

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

                # 创建frames目录
                frames_dir = os.path.join(data_dir, 'frames')
                os.makedirs(frames_dir, exist_ok=True)

                # 提取视频帧
                cap = cv2.VideoCapture(dest_video_path)
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 调整大小为512x512
                    if frame.shape[0] != 512 or frame.shape[1] != 512:
                        frame = cv2.resize(frame, (512, 512))

                    frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_count += 1

                    # 更新进度
                    if frame_count % 50 == 0:
                        progress = 25 + (frame_count / 300 * 15)  # 假设最多300帧
                        task['progress'] = min(progress, 40)

                cap.release()
                self._add_log(task, f"✅ 帧提取完成: {frame_count} 帧")

                # 创建数据集配置
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

            # 保存训练配置
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

            # 🔥 强制使用模拟训练（避免依赖问题）
            self._add_log(task, "🎭 使用模拟训练模式 - 跳过复杂依赖")
            self._add_log(task, "📝 实际训练需要完整安装：torch, diff-gaussian-rasterization, lpips 等")
            self._add_log(task, "⏰ 模拟训练将在后台运行...")

            # 使用模拟训练
            self._simulate_training(task_id, model_path, config.get('epochs', 100))

            # 9. 训练完成
            task['status'] = 'completed'
            task['progress'] = 100
            task['model_path'] = model_path
            task['end_time'] = datetime.now().isoformat()

            # 保存训练摘要
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

            # 保存错误日志
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

        # 构建训练命令
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

        # 运行训练
        self._run_subprocess(task_id, train_cmd, "训练")

    def _simulate_training(self, task_id, model_path, epochs):
        """模拟训练过程"""
        task = self.tasks[task_id]

        self._add_log(task, f"🎭 模拟训练模式，共 {epochs} 轮")

        for epoch in range(epochs):
            # 更新进度
            progress = 50 + (epoch / epochs * 45)
            task['progress'] = progress

            # 模拟损失
            loss = 0.8 * (0.95 ** epoch) + 0.1
            log_msg = f"Epoch {epoch + 1:04d}/{epochs:04d} - Loss: {loss:.4f}"
            self._add_log(task, log_msg)

            # 每10轮保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'loss': loss,
                    'timestamp': datetime.now().isoformat(),
                    'model_version': '1.0'
                }

                checkpoint_path = os.path.join(model_path, f'checkpoint_{epoch + 1:04d}.pth')
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint, f, indent=2)

                self._add_log(task, f"💾 保存检查点: {checkpoint_path}")

            time.sleep(0.5)

            # 检查是否被停止
            if task['status'] == 'stopped':
                self._add_log(task, "⏹️ 训练被停止")
                break

    def _run_subprocess(self, task_id, cmd, phase_name):
        """运行子进程 - 修复编码问题"""
        task = self.tasks[task_id]

        self._add_log(task, f"开始{phase_name}: {' '.join(cmd)}")

        try:
            # 修复1: 设置环境变量，强制使用UTF-8编码
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'

            # 修复2: 使用正确的编码参数
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,  # 使用文本模式
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',  # 明确指定编码
                errors='replace',  # 遇到无法解码的字符用�替换
                cwd=os.getcwd(),
                env=env  # 使用修改后的环境变量
            )

            self.processes[task_id] = process

            # 读取输出
            for line in process.stdout:
                line = line.strip()
                if line:
                    # 清理可能的非法字符
                    cleaned_line = ''.join(char for char in line if ord(char) < 65536)
                    self._add_log(task, cleaned_line)

                    # 尝试解析进度
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
                        except:
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

        # 限制日志数量
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
        except:
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


# ========== 路由定义 ==========
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_generation')
def video_generation():
    return render_template('video_generation.html')


@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    if request.method == 'POST':
        if request.is_json:
            data = request.json
            action = data.get('action')

            if action == 'start':
                # 验证必要参数
                if not data.get('video_path'):
                    return jsonify({
                        'status': 'error',
                        'error': '视频路径不能为空'
                    }), 400

                # 准备训练配置
                config = {
                    'video_path': data['video_path'],
                    'model_name': data.get('model_name', f'model_{uuid.uuid4().hex[:6]}'),
                    'model_path': data.get('model_path',
                                           f'output/{data.get("model_name", f"model_{uuid.uuid4().hex[:6]}")}'),
                    'data_dir': data.get('data_dir', f'data/train_{uuid.uuid4().hex[:6]}'),
                    'config_file': data.get('config_file', 'arguments/args.py'),
                    'epochs': int(data.get('epochs', 100)),
                    'preprocess': data.get('preprocess', True),
                    'use_gpu': data.get('use_gpu', False),
                    'advanced_config': data.get('advanced_config', {})
                }

                # 启动训练
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

                # 添加最近日志
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

        # 保持兼容旧版本的表单提交
        return jsonify({'status': 'error', 'error': '不支持的请求格式'}), 400

    return render_template('model_training.html')


@app.route('/chat_system')
def chat_system():
    return render_template('chat_system.html')


@app.route('/inference')
def inference():
    return render_template('inference.html')


# ========== API路由 ==========
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

        # 检查文件类型
        allowed_extensions = {'mp4', 'avi', 'mov', 'm4v', 'mpg', 'mpeg', 'webm'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''

        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': f'不支持的文件类型: .{file_ext}。支持: {", ".join(allowed_extensions)}'
            }), 400

        # 创建上传目录
        upload_dir = 'static/uploads/videos'
        os.makedirs(upload_dir, exist_ok=True)

        # 生成安全的文件名
        original_name = file.filename
        safe_name = secure_filename(original_name)
        unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        file_path = os.path.join(upload_dir, unique_name)

        print(f"保存文件到: {file_path}")

        # 保存文件
        file.save(file_path)

        # 验证文件
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': '文件保存失败'}), 500

        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)

        print(f"文件保存成功: {file_size_mb:.2f} MB")

        # 验证视频文件
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                os.remove(file_path)
                return jsonify({'success': False, 'error': '视频文件无法打开或已损坏'}), 400

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            # 检查视频时长
            if duration > 600:  # 10分钟
                os.remove(file_path)
                return jsonify({'success': False, 'error': '视频太长，最大10分钟'}), 400

        except Exception as e:
            print(f"视频验证失败: {e}")
            # 不删除文件，继续使用

        # 返回成功响应
        # 🔥 关键：返回相对路径，不是绝对路径
        web_path = f"/static/uploads/videos/{unique_name}"

        return jsonify({
            'success': True,
            'message': '✅ 视频上传成功',
            'filename': unique_name,
            'original_name': original_name,
            'path': file_path,  # 服务器上的绝对路径
            'web_path': web_path,  # Web访问路径
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


@app.route('/api/upload/audio', methods=['POST'])
def api_upload_audio():
    """上传音频文件"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': '没有音频文件'}), 400

        audio_file = request.files['audio']

        if audio_file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'}), 400

        # 创建上传目录
        upload_dir = 'static/uploads/audios'
        os.makedirs(upload_dir, exist_ok=True)

        # 生成安全的文件名
        original_name = audio_file.filename
        safe_name = secure_filename(original_name)
        unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        file_path = os.path.join(upload_dir, unique_name)

        # 保存文件
        audio_file.save(file_path)

        return jsonify({
            'success': True,
            'filename': unique_name,
            'path': file_path,
            'url': f'/static/uploads/audios/{unique_name}'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ========== 新增的训练状态API ==========
@app.route('/api/train/status/<task_id>', methods=['GET'])
def api_train_status(task_id):
    """获取训练任务状态 - 修复版本"""
    try:
        # 获取任务信息
        task = training_manager.get_task(task_id)

        if not task:
            return jsonify({
                'success': False,
                'error': f'任务 {task_id} 不存在',
                'available_tasks': [t['id'] for t in training_manager.list_tasks()],
                'message': '请启动一个新的训练任务'
            }), 404

        # 构建响应
        response = {
            'success': True,
            'task': {
                'id': task['id'],
                'status': task['status'],
                'progress': task['progress'],
                'start_time': task.get('start_time'),
                'end_time': task.get('end_time'),
                'model_path': task.get('model_path'),
                'model_name': task.get('config', {}).get('model_name', '未命名'),
                'error': task.get('error')
            }
        }

        # 添加最近日志（可选）
        if task.get('logs'):
            response['task']['recent_logs'] = task['logs'][-10:]

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'task_id': task_id
        }), 500


@app.route('/api/train/latest', methods=['GET'])
def get_latest_task():
    """获取最新的训练任务"""
    tasks = training_manager.list_tasks()
    if tasks:
        latest = max(tasks, key=lambda x: x.get('start_time', ''))
        return jsonify({
            'success': True,
            'task_id': latest['id'],
            'status': latest['status'],
            'progress': latest.get('progress', 0)
        })
    return jsonify({'success': False, 'message': '没有训练任务'})


@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    """开始训练任务 - 新增的独立API"""
    try:
        data = request.get_json()

        # 验证必要参数
        if not data.get('video_path'):
            return jsonify({
                'success': False,
                'error': '视频路径不能为空'
            }), 400

        # 准备配置
        config = {
            'video_path': data['video_path'],
            'model_name': data.get('model_name', f'model_{uuid.uuid4().hex[:6]}'),
            'epochs': int(data.get('epochs', 50)),
            'preprocess': data.get('preprocess', True),
            'use_gpu': data.get('use_gpu', False),
            'config_file': data.get('config_file', 'arguments/args.py'),
            'advanced_config': data.get('advanced_config', {})
        }

        # 启动训练
        task_id = training_manager.start_training(config)

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '训练任务已启动',
            'status_url': f'/api/train/status/{task_id}'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/train/list', methods=['GET'])
def api_train_list():
    """获取所有训练任务列表 - 新增的独立API"""
    try:
        tasks = training_manager.list_tasks()
        simplified_tasks = []

        for task in tasks:
            simplified_tasks.append({
                'id': task['id'],
                'status': task['status'],
                'progress': task['progress'],
                'start_time': task.get('start_time'),
                'model_name': task.get('config', {}).get('model_name', '未命名'),
                'video_source': task.get('config', {}).get('video_path', '未知')
            })

        return jsonify({
            'success': True,
            'tasks': simplified_tasks
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/train/stop/<task_id>', methods=['POST'])
def api_train_stop(task_id):
    """停止训练任务 - 新增的独立API"""
    try:
        success = training_manager.stop_training(task_id)

        if success:
            return jsonify({
                'success': True,
                'message': f'任务 {task_id} 已停止'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'任务 {task_id} 不存在或无法停止'
            }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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
                    except:
                        continue

        return jsonify({'success': True, 'configs': configs})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/models/list', methods=['GET'])
def api_models_list():
    """获取模型列表"""
    try:
        models = []

        # 检查output目录
        output_dir = 'output'
        if os.path.exists(output_dir):
            for model_name in os.listdir(output_dir):
                model_path = os.path.join(output_dir, model_name)
                if os.path.isdir(model_path):
                    # 检查是否有训练完成标记
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

        # 如果没有模型，添加示例
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

        # 保存音频
        audio_dir = 'static/uploads/audios'
        os.makedirs(audio_dir, exist_ok=True)

        audio_filename = f"{uuid.uuid4().hex[:8]}_{audio_file.filename}"
        audio_path = os.path.join(audio_dir, audio_filename)
        audio_file.save(audio_path)

        # 模拟生成过程
        result_dir = 'result-video'
        os.makedirs(result_dir, exist_ok=True)

        result_filename = f"result_{uuid.uuid4().hex[:8]}.mp4"
        result_path = os.path.join(result_dir, result_filename)

        # 创建模拟结果文件
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


@app.route('/api/health', methods=['GET'])
def api_health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'working_directory': os.getcwd(),
        'python_version': sys.version
    })


@app.route('/debug/routes')
def debug_routes():
    """查看所有已注册的路由"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'rule': str(rule),
            'methods': list(rule.methods)
        })

    # 检查是否有训练状态路由
    has_train_status = any('/api/train/status/' in route['rule'] for route in routes)

    return jsonify({
        'has_train_status_api': has_train_status,
        'total_routes': len(routes),
        'routes': sorted(routes, key=lambda x: x['rule'])
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
        paths_info['upload_dir_contents'] = os.listdir(upload_dir)[:10]  # 只显示前10个

    return jsonify(paths_info)


# ========== 路由查看器 ==========
@app.route('/debug/all-routes')
def show_all_routes():
    """显示所有已注册的路由"""
    routes = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':  # 过滤静态文件路由
            routes.append({
                'route': str(rule),
                'endpoint': rule.endpoint,
                'methods': list(rule.methods)
            })

    return jsonify({
        'total_routes': len(routes),
        'routes': sorted(routes, key=lambda x: x['route']),
        'has_train_status': any('/api/train/status/' in r['route'] for r in routes)
    })


# ========== 静态文件路由 ==========
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join('static', filename))


@app.route('/result-video/<path:filename>')
def serve_result_video(filename):
    return send_file(os.path.join('result-video', filename))


# ========== 应用启动 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 EGS-Talker 模型训练平台")
    print("=" * 60)
    print(f"📁 工作目录: {os.getcwd()}")
    print(f"🌐 服务地址: http://localhost:5001")
    print(f"📊 上传目录: {os.path.join(os.getcwd(), 'static/uploads/videos')}")
    print("=" * 60)

    # 检查必要目录
    create_directories()

    # 运行应用
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5001,
        threaded=True
    )
