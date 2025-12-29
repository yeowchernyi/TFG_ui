from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import uuid
import threading
import subprocess
import signal
from datetime import datetime
from backend.video_generator import generate_video
from backend.model_trainer import train_model
from backend.chat_engine import chat_response

app = Flask(__name__)

# 训练任务管理器类
class TrainingManager:
    def __init__(self):
        self.tasks = {}
        self.processes = {}
    
    def start_training(self, config):
        """启动训练任务"""
        task_id = str(uuid.uuid4())
        
        # 保存配置
        self.tasks[task_id] = {
            'id': task_id,
            'config': config,
            'status': 'starting',
            'progress': 0,
            'logs': [],
            'start_time': datetime.now().isoformat(),
            'model_path': None
        }
        
        # 在后台线程中运行训练
        thread = threading.Thread(
            target=self._run_training_task,
            args=(task_id, config)
        )
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def _run_training_task(self, task_id, config):
        """执行训练任务"""
        task = self.tasks[task_id]
        
        try:
            task['status'] = 'preprocessing'
            task['progress'] = 10
            
            # 1. 数据预处理（如果需要）
            if config.get('preprocess_video') and config.get('video_path'):
                video_path = config['video_path']
                data_dir = config.get('data_dir', f"data/training_{task_id}")
                
                # 确保数据目录存在
                os.makedirs(data_dir, exist_ok=True)
                
                # 运行预处理
                preprocess_cmd = [
                    sys.executable, 'data_utils/process.py',
                    video_path, '--output_dir', data_dir
                ]
                
                self._run_subprocess(task_id, preprocess_cmd, "预处理")
                task['progress'] = 30
            
            # 2. 训练模型
            task['status'] = 'training'
            data_path = config.get('data_dir', f"data/training_{task_id}")
            model_path = config.get('model_path', f"output/training_{task_id}")
            
            # 确保输出目录存在
            os.makedirs(model_path, exist_ok=True)
            
            # 构建训练命令 - 根据你的train.py参数调整
            train_cmd = [
                sys.executable, 'train.py',
                '-s', data_path,
                '--model_path', model_path
            ]
            
            # 添加配置文件参数
            if config.get('config_file'):
                train_cmd.extend(['--configs', config['config_file']])
            
            # 添加可选参数
            if config.get('epochs'):
                train_cmd.extend(['--epochs', str(config['epochs'])])
            
            # 运行训练
            self._run_subprocess(task_id, train_cmd, "训练")
            
            # 训练成功
            task['status'] = 'completed'
            task['progress'] = 100
            task['model_path'] = model_path
            task['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
            task['logs'].append(f"错误: {str(e)}")
            task['end_time'] = datetime.now().isoformat()
    
    def _run_subprocess(self, task_id, cmd, phase_name):
        """运行子进程并捕获输出"""
        task = self.tasks[task_id]
        
        # 记录命令
        task['logs'].append(f"开始{phase_name}: {' '.join(cmd)}")
        
        # 运行进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 保存进程引用（用于停止）
        self.processes[task_id] = process
        
        # 实时读取输出
        for line in process.stdout:
            log_line = line.strip()
            task['logs'].append(log_line)
            
            # 更新进度（根据实际日志解析）
            if 'Epoch' in log_line and '/' in log_line:
                try:
                    # 解析进度，例如: Epoch 10/100
                    parts = log_line.split('Epoch')[1].strip().split('/')
                    current = int(parts[0].strip())
                    total = int(parts[1].split()[0].strip())
                    
                    # 计算进度百分比
                    if phase_name == "训练":
                        base_progress = 30  # 预处理完成后
                        progress_range = 70  # 训练占70%
                        task['progress'] = base_progress + (progress_range * current / total)
                except:
                    pass
            
            # 限制日志数量
            if len(task['logs']) > 1000:
                task['logs'] = task['logs'][-1000:]
        
        # 等待进程完成
        process.wait()
        
        # 移除进程引用
        if task_id in self.processes:
            del self.processes[task_id]
        
        if process.returncode != 0:
            raise Exception(f"{phase_name}失败，退出码: {process.returncode}")
    
    def stop_training(self, task_id):
        """停止训练任务"""
        if task_id in self.processes:
            process = self.processes[task_id]
            
            # 终止进程
            if os.name == 'nt':  # Windows
                process.terminate()
            else:  # Unix/Linux/Mac
                os.kill(process.pid, signal.SIGTERM)
            
            # 更新任务状态
            if task_id in self.tasks:
                self.tasks[task_id]['status'] = 'stopped'
                self.tasks[task_id]['logs'].append("训练已被用户停止")
            
            # 清理进程引用
            del self.processes[task_id]
            
            return True
        return False
    
    def get_task(self, task_id):
        """获取任务信息"""
        return self.tasks.get(task_id)
    
    def list_tasks(self):
        """列出所有任务"""
        return list(self.tasks.values())

# 初始化训练管理器
training_manager = TrainingManager()

# 首页
@app.route('/')
def index():
    return render_template('index.html')

# 视频生成界面
@app.route('/video_generation', methods=['GET', 'POST'])
def video_generation():
    if request.method == 'POST':
        data = {
            "model_name": request.form.get('model_name'),
            "model_param": request.form.get('model_param'),
            "ref_audio": request.form.get('ref_audio'),
            "gpu_choice": request.form.get('gpu_choice'),
            "target_text": request.form.get('target_text'),
        }

        video_path = generate_video(data)
        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('video_generation.html')

# 模型训练界面 - 修改后的版本
@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    if request.method == 'POST':
        # 检查是否是训练启动请求
        if request.is_json:
            data = request.json
            action = data.get('action')
            
            if action == 'start':
                # 训练启动请求
                config = {
                    'video_path': data.get('video_path'),
                    'data_dir': data.get('data_dir'),
                    'model_path': data.get('model_path', f"output/{uuid.uuid4().hex[:8]}"),
                    'config_file': data.get('config_file', 'arguments/args.py'),
                    'epochs': int(data.get('epochs', 100)),
                    'preprocess_video': data.get('preprocess', True)
                }
                
                task_id = training_manager.start_training(config)
                return jsonify({
                    'status': 'success',
                    'task_id': task_id,
                    'message': '训练任务已启动'
                })
            
            elif action == 'status':
                # 获取训练状态请求
                task_id = data.get('task_id')
                task = training_manager.get_task(task_id)
                
                if not task:
                    return jsonify({
                        'status': 'error',
                        'error': '任务不存在'
                    }), 404
                
                response = {
                    'status': 'success',
                    'task': {
                        'id': task['id'],
                        'status': task['status'],
                        'progress': task['progress'],
                        'start_time': task.get('start_time'),
                        'end_time': task.get('end_time'),
                        'model_path': task.get('model_path')
                    }
                }
                
                # 添加日志（只返回最近20条）
                if task.get('logs'):
                    response['task']['recent_logs'] = task['logs'][-20:]
                
                # 添加错误信息
                if task['status'] == 'failed' and 'error' in task:
                    response['task']['error'] = task['error']
                
                return jsonify(response)
            
            elif action == 'stop':
                # 停止训练请求
                task_id = data.get('task_id')
                success = training_manager.stop_training(task_id)
                
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': '训练任务已停止'
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'error': '任务不存在或无法停止'
                    }), 404
            
            elif action == 'list':
                # 获取任务列表
                tasks = training_manager.list_tasks()
                simplified_tasks = []
                
                for task in tasks:
                    simplified_tasks.append({
                        'id': task['id'],
                        'status': task['status'],
                        'progress': task['progress'],
                        'start_time': task.get('start_time'),
                        'model_path': task.get('model_path')
                    })
                
                return jsonify({
                    'status': 'success',
                    'tasks': simplified_tasks
                })
        
        # 原有的训练表单提交（保持兼容性）
        data = {
            "model_choice": request.form.get('model_choice'),
            "ref_video": request.form.get('ref_video'),
            "gpu_choice": request.form.get('gpu_choice'),
            "epoch": request.form.get('epoch'),
            "custom_params": request.form.get('custom_params')
        }

        video_path = train_model(data)
        video_path = "/" + video_path.replace("\\", "/")

        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('model_training.html')

# 训练API端点
@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    """启动训练API"""
    try:
        data = request.json
        
        # 验证必要参数
        if not data.get('video_path') and not data.get('data_dir'):
            return jsonify({
                'success': False,
                'error': '必须提供视频路径或数据目录'
            }), 400
        
        # 训练配置
        config = {
            'video_path': data.get('video_path'),
            'data_dir': data.get('data_dir'),
            'model_path': data.get('model_path', f"output/{uuid.uuid4().hex[:8]}"),
            'config_file': data.get('config_file', 'arguments/args.py'),
            'epochs': data.get('epochs', 100),
            'preprocess_video': data.get('preprocess', True)
        }
        
        # 启动训练
        task_id = training_manager.start_training(config)
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '训练任务已启动'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/train/status/<task_id>', methods=['GET'])
def api_train_status(task_id):
    """获取训练状态API"""
    task = training_manager.get_task(task_id)
    
    if not task:
        return jsonify({
            'success': False,
            'error': '任务不存在'
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
            'model_path': task.get('model_path')
        }
    }
    
    # 添加日志（只返回最近50条）
    if task.get('logs'):
        response['task']['recent_logs'] = task['logs'][-50:]
    
    # 添加错误信息
    if task['status'] == 'failed' and 'error' in task:
        response['task']['error'] = task['error']
    
    return jsonify(response)

@app.route('/api/train/list', methods=['GET'])
def api_train_list():
    """获取训练任务列表API"""
    tasks = training_manager.list_tasks()
    
    # 简化的任务信息
    simplified_tasks = []
    for task in tasks:
        simplified_tasks.append({
            'id': task['id'],
            'status': task['status'],
            'progress': task['progress'],
            'start_time': task.get('start_time'),
            'model_path': task.get('model_path')
        })
    
    return jsonify({
        'success': True,
        'tasks': simplified_tasks
    })

@app.route('/api/train/stop/<task_id>', methods=['POST'])
def api_train_stop(task_id):
    """停止训练任务API"""
    success = training_manager.stop_training(task_id)
    
    if success:
        return jsonify({
            'success': True,
            'message': '训练任务已停止'
        })
    else:
        return jsonify({
            'success': False,
            'error': '任务不存在或无法停止'
        }), 404

@app.route('/api/upload/video', methods=['POST'])
def api_upload_video():
    """上传视频文件API"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'}), 400
        
        # 创建上传目录
        upload_dir = 'static/uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        # 生成唯一文件名
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # 保存文件
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'path': file_path,
            'url': f'/static/uploads/{unique_filename}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# 实时对话系统界面
@app.route('/chat_system', methods=['GET', 'POST'])
def chat_system():
    if request.method == 'POST':
        data = {
            "model_name": request.form.get('model_name'),
            "model_param": request.form.get('model_param'),
            "voice_clone": request.form.get('voice_clone'),
            "api_choice": request.form.get('api_choice'),
        }

        video_path = chat_response(data)
        video_path = "/" + video_path.replace("\\", "/")

        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('chat_system.html')

@app.route('/save_audio', methods=['POST'])
def save_audio():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': '没有音频文件'})
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})
    
    # 确保目录存在
    os.makedirs('./static/audios', exist_ok=True)
    
    # 保存文件
    audio_file.save('./static/audios/input.wav')
    
    return jsonify({'status': 'success', 'message': '音频保存成功'})

if __name__ == '__main__':
    # 确保必要的目录存在
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    app.run(debug=True, port=5001)
