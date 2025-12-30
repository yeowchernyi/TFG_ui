from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import uuid
import json
import threading
import subprocess
import shutil
from datetime import datetime
from werkzeug.utils import secure_filename
import signal

# 导入现有的后端模块
try:
    from backend.video_generator import generate_video
    from backend.model_trainer import train_model
    from backend.chat_engine import chat_response
except ImportError:
    # 如果导入失败，创建虚拟函数
    def generate_video(data):
        return "static/videos/out.mp4"
    
    def train_model(data):
        return "static/videos/demo.mp4"
    
    def chat_response(data):
        return "static/videos/chat.mp4"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB文件上传限制

# ========== 训练任务管理器 ==========
class TrainingManager:
    def __init__(self):
        self.tasks = {}
        self.processes = {}
    
    def start_training(self, config):
        """启动训练任务"""
        task_id = str(uuid.uuid4())
        
        self.tasks[task_id] = {
            'id': task_id,
            'config': config,
            'status': 'starting',
            'progress': 0,
            'logs': [],
            'start_time': datetime.now().isoformat(),
            'model_path': None
        }
        
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
            
            # 1. 获取视频路径
            video_path = config.get('video_path')
            if not video_path or not os.path.exists(video_path):
                raise Exception(f"视频文件不存在: {video_path}")
            
            # 2. 创建数据目录
            data_dir = config.get('data_dir', f"data/training_{task_id}")
            os.makedirs(data_dir, exist_ok=True)
            
            # 3. 预处理（如果需要）
            if config.get('preprocess_video', True):
                self._add_log(task, "开始视频预处理...")
                preprocess_cmd = [
                    sys.executable, 'data_utils/process.py',
                    video_path, '--output_dir', data_dir
                ]
                self._run_subprocess(task_id, preprocess_cmd, "预处理")
                task['progress'] = 30
            
            # 4. 训练模型
            task['status'] = 'training'
            model_path = config.get('model_path', f"output/training_{task_id}")
            os.makedirs(model_path, exist_ok=True)
            
            # 构建训练命令
            train_cmd = [
                sys.executable, 'train.py',
                '-s', data_dir,
                '--model_path', model_path
            ]
            
            if config.get('config_file'):
                train_cmd.extend(['--configs', config['config_file']])
            if config.get('epochs'):
                train_cmd.extend(['--epochs', str(config['epochs'])])
            
            # 应用高级配置
            if 'advanced_config' in config:
                config_path = os.path.join(model_path, 'config.json')
                with open(config_path, 'w') as f:
                    json.dump(config['advanced_config'], f, indent=2)
                self._add_log(task, f"高级配置已保存: {config_path}")
            
            self._run_subprocess(task_id, train_cmd, "训练")
            
            # 训练成功
            task['status'] = 'completed'
            task['progress'] = 100
            task['model_path'] = model_path
            task['end_time'] = datetime.now().isoformat()
            self._add_log(task, "🎉 训练完成！")
            
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
            self._add_log(task, f"❌ 训练失败: {str(e)}")
            task['end_time'] = datetime.now().isoformat()
    
    def _run_subprocess(self, task_id, cmd, phase_name):
        """运行子进程"""
        task = self.tasks[task_id]
        self._add_log(task, f"开始{phase_name}: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.processes[task_id] = process
        
        for line in process.stdout:
            self._add_log(task, line.strip())
            
            # 解析进度
            if 'Epoch' in line and '/' in line:
                try:
                    parts = line.split('Epoch')[1].strip().split('/')
                    current = int(parts[0].strip())
                    total = int(parts[1].split()[0].strip())
                    base = 30 if phase_name == "训练" else 0
                    task['progress'] = base + (70 * current / total)
                except:
                    pass
        
        process.wait()
        
        if task_id in self.processes:
            del self.processes[task_id]
        
        if process.returncode != 0:
            raise Exception(f"{phase_name}失败，退出码: {process.returncode}")
    
    def _add_log(self, task, message):
        """添加日志"""
        task['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        if len(task['logs']) > 1000:
            task['logs'] = task['logs'][-1000:]
    
    def stop_training(self, task_id):
        """停止训练"""
        if task_id in self.processes:
            process = self.processes[task_id]
            if os.name == 'nt':
                process.terminate()
            else:
                os.kill(process.pid, signal.SIGTERM)
            
            if task_id in self.tasks:
                self.tasks[task_id]['status'] = 'stopped'
                self.tasks[task_id]['logs'].append("训练已被用户停止")
            
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

@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    if request.method == 'POST':
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
                    'preprocess_video': data.get('preprocess', True),
                    'advanced_config': data.get('advanced_config', {})
                }
                
                task_id = training_manager.start_training(config)
                return jsonify({
                    'status': 'success',
                    'task_id': task_id,
                    'message': '训练任务已启动'
                })
            
            elif action == 'status':
                # 获取训练状态
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
                        'model_path': task.get('model_path')
                    }
                }
                
                if task.get('logs'):
                    response['task']['recent_logs'] = task['logs'][-20:]
                
                if task['status'] == 'failed' and 'error' in task:
                    response['task']['error'] = task['error']
                
                return jsonify(response)
            
            elif action == 'stop':
                # 停止训练
                task_id = data.get('task_id')
                success = training_manager.stop_training(task_id)
                
                if success:
                    return jsonify({'status': 'success', 'message': '训练任务已停止'})
                else:
                    return jsonify({'status': 'error', 'error': '任务不存在或无法停止'}), 404
            
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
                
                return jsonify({'status': 'success', 'tasks': simplified_tasks})
        
        # 原有的表单提交（保持兼容）
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

# ========== API路由 ==========
@app.route('/api/upload/video', methods=['POST'])
def api_upload_video():
    """上传视频文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'}), 400
        
        upload_dir = 'static/uploads/videos'
        os.makedirs(upload_dir, exist_ok=True)
        
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'path': file_path,
            'url': f'/static/uploads/videos/{unique_filename}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    """启动训练API"""
    try:
        data = request.json
        
        if not data.get('video_path') and not data.get('data_dir'):
            return jsonify({'success': False, 'error': '必须提供视频路径或数据目录'}), 400
        
        config = {
            'video_path': data.get('video_path'),
            'data_dir': data.get('data_dir'),
            'model_path': data.get('model_path', f"output/{uuid.uuid4().hex[:8]}"),
            'config_file': data.get('config_file', 'arguments/args.py'),
            'epochs': data.get('epochs', 100),
            'preprocess_video': data.get('preprocess', True)
        }
        
        task_id = training_manager.start_training(config)
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '训练任务已启动'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train/status/<task_id>', methods=['GET'])
def api_train_status(task_id):
    """获取训练状态API"""
    task = training_manager.get_task(task_id)
    
    if not task:
        return jsonify({'success': False, 'error': '任务不存在'}), 404
    
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
    
    if task.get('logs'):
        response['task']['recent_logs'] = task['logs'][-50:]
    
    if task['status'] == 'failed' and 'error' in task:
        response['task']['error'] = task['error']
    
    return jsonify(response)

@app.route('/api/train/list', methods=['GET'])
def api_train_list():
    """获取训练任务列表API"""
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
    
    return jsonify({'success': True, 'tasks': simplified_tasks})

@app.route('/api/train/stop/<task_id>', methods=['POST'])
def api_train_stop(task_id):
    """停止训练任务API"""
    success = training_manager.stop_training(task_id)
    
    if success:
        return jsonify({'success': True, 'message': '训练任务已停止'})
    else:
        return jsonify({'success': False, 'error': '任务不存在或无法停止'}), 404

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
        
        # 保存JSON
        config_path = os.path.join(config_dir, f'{config_name}.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # 保存Python
        py_config_path = os.path.join(config_dir, f'{config_name}.py')
        with open(py_config_path, 'w', encoding='utf-8') as f:
            f.write(f"ModelHiddenParams = {json.dumps(config_data.get('ModelHiddenParams', {}), indent=4)}\n\n")
            f.write(f"OptimizationParams = {json.dumps(config_data.get('OptimizationParams', {}), indent=4)}\n")
        
        return jsonify({
            'success': True,
            'message': f'配置已保存',
            'paths': {'json': config_path, 'python': py_config_path}
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
                            'size': os.path.getsize(config_path),
                            'modified': datetime.fromtimestamp(os.path.getmtime(config_path)).strftime('%Y-%m-%d %H:%M'),
                            'config': config_data
                        })
                    except:
                        pass
        
        return jsonify({'success': True, 'configs': configs})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/inference')
def inference_page():
    """推理页面"""
    return render_template('inference.html')

@app.route('/api/models/list')
def api_models_list():
    """获取模型列表"""
    try:
        models = []
        sync_talk_model_dir = 'SyncTalk/model'
        
        if os.path.exists(sync_talk_model_dir):
            for model_name in os.listdir(sync_talk_model_dir):
                model_path = os.path.join(sync_talk_model_dir, model_name)
                
                if os.path.isdir(model_path):
                    # 检查是否有模型文件
                    model_files = []
                    for ext in ['.pth', '.pkl', '.ckpt', '.bin', '.pt']:
                        model_files.extend([f for f in os.listdir(model_path) if f.endswith(ext)])
                    
                    if model_files:
                        try:
                            created_time = os.path.getctime(model_path)
                            created_str = datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M')
                        except:
                            created_str = '未知时间'
                        
                        # 计算大小
                        total_size = 0
                        for root, dirs, files in os.walk(model_path):
                            for file in files:
                                total_size += os.path.getsize(os.path.join(root, file))
                        
                        models.append({
                            'name': model_name,
                            'display_name': f"SyncTalk - {model_name}",
                            'path': model_path,
                            'created_date': created_str,
                            'size': f"{total_size / 1024 / 1024:.1f} MB",
                            'file_count': len(model_files),
                            'type': 'SyncTalk模型'
                        })
        
        # 检查其他目录
        other_dirs = ['output', 'checkpoints', 'pretrained_models']
        for model_dir in other_dirs:
            if os.path.exists(model_dir):
                for item in os.listdir(model_dir):
                    item_path = os.path.join(model_dir, item)
                    
                    if os.path.isdir(item_path):
                        model_files = []
                        for ext in ['.pth', '.pt', '.ckpt', '.pkl', '.bin']:
                            model_files.extend([f for f in os.listdir(item_path) if f.endswith(ext)])
                        
                        if model_files:
                            models.append({
                                'name': item,
                                'display_name': f"{model_dir} - {item}",
                                'path': item_path,
                                'created_date': '未知',
                                'size': '未知',
                                'file_count': len(model_files),
                                'type': '训练模型'
                            })
        
        if not models:
            models.append({
                'name': '无可用模型',
                'display_name': '暂无可用模型',
                'path': '',
                'created_date': '-',
                'size': '0 MB',
                'file_count': 0,
                'type': '信息',
                'instructions': '请先训练模型或下载预训练模型'
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
        
        import time
        start_time = time.time()
        
        try:
            # 使用现有的 video_generator
            from backend.video_generator import generate_video
            
            inference_data = {
                'model_name': 'SyncTalk',
                'model_param': model_path,
                'ref_audio': audio_path,
                'gpu_choice': '0',
                'target_text': ''
            }
            
            # 添加渲染参数
            chunk_size = request.form.get('chunk_size', '0')
            if int(chunk_size) > 0:
                inference_data['chunk_size'] = chunk_size
            
            result_path = generate_video(inference_data)
            
        except ImportError:
            raise Exception("无法导入推理模块")
        
        # 验证结果
        if not os.path.exists(result_path):
            raise Exception(f"结果文件未找到: {result_path}")
        
        processing_time = time.time() - start_time
        file_size = os.path.getsize(result_path) if os.path.exists(result_path) else 0
        
        # 确保Web可访问路径
        web_path = result_path.replace('\\', '/')
        if not web_path.startswith('/'):
            web_path = '/' + web_path
        
        return jsonify({
            'success': True,
            'video_url': web_path,
            'filename': os.path.basename(result_path),
            'size': f"{file_size / 1024 / 1024:.2f} MB",
            'processing_time': f"{processing_time:.2f}",
            'message': '视频生成成功'
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'details': traceback.format_exc()
        }), 500

@app.route('/save_audio', methods=['POST'])
def save_audio():
    """保存音频"""
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': '没有音频文件'})
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})
    
    os.makedirs('./static/audios', exist_ok=True)
    audio_file.save('./static/audios/input.wav')
    
    return jsonify({'status': 'success', 'message': '音频保存成功'})

if __name__ == '__main__':
    # 确保目录存在
    os.makedirs('static/uploads/videos', exist_ok=True)
    os.makedirs('static/uploads/audios', exist_ok=True)
    os.makedirs('static/videos', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
