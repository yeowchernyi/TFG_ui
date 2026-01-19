# EGSTalker UI：数字人生成与训练一体化平台

本仓库整合了 **EGSTalker 三维高斯变形渲染**、**Qwen 大语言模型对话** 与 **Edge-TTS 语音合成**，提供一站式网页交互（Flask）用于文本/音频驱动的数字人视频生成，并支持训练、推理与资源管理。

---

## 1. 服务器与资源规范
- 服务器地址：10.108.17.241，端口 1986，账号 xsj，密码为 by8h7NY7BQOsw1Ff。
- GPU 使用：每人 1 块（短时可 2–4 块，<24h），GPU6/7 为博士专用；温度需 <88°C，使用 `nvtop` 监控；使用 `export CUDA_VISIBLE_DEVICES=<id>` 绑定。
- 存储：个人不超过 200GB，常用命令 `df -h`、`du -sh`，及时清理。
- 监控：`nvtop` 查看 GPU，`htop` 查看 CPU/内存；端口冲突时更换 5001–5100。

---

## 2. 环境准备
建议使用 Conda，已提供多份环境文件，按需选择：

- `egstalker.yml` / `egstalker_py39.yml`：主环境，运行 UI、LLM、渲染后端。
- `egstalker_track_py39.yml`：人脸/三维追踪依赖。
- `egstalker_tf_py39.yml`：DeepSpeech 特征提取（如需）。

### 快速创建主环境（推荐）
```bash
cd ~/projects/egstalker/TFG_ui
conda env create -f egstalker.yml -n egstalker_py39
conda activate egstalker_py39
```

若失败，可手动：
```bash
conda create -n egstalker_py39 python=3.9 -y
conda activate egstalker_py39
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install Flask==3.0.3 opencv-python numpy scipy requests edge-tts pydub
```

验证 CUDA：
```bash
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
PY
```

---

## 3. 目录结构速览
- `app.py`：Flask 主入口，页面路由与 API。
- `backend/`：Qwen 文本对话、训练管理、视频生成逻辑。
- `EGSTalker_Model/`：渲染后端 Flask 服务，接收音频生成视频（默认 5002）。
- `templates/`、`static/`：前端页面与静态资源。
- `arguments/`、`configs/`：训练/推理参数配置。
- `data/`、`output/`、`result-video/`：数据与输出存储。
- `scene/`、`render.py`、`infer.py`：底层渲染与推理工具。

---

## 4. 运行应用（最小可用流程）
需同时启动两个进程：

### 4.1 启动渲染后端（终端 1）
```bash
conda activate egstalker_py39
cd ~/projects/egstalker/TFG_ui
export CUDA_VISIBLE_DEVICES=0    # 选择空闲 GPU
python app.py                    # 监听 http://0.0.0.0:5002
```

### 4.2 启动主 UI（终端 2）
```bash
conda activate egstalker_py39
cd ~/projects/egstalker/TFG_ui
export CUDA_VISIBLE_DEVICES=0
python app.py                    # 监听 http://0.0.0.0:5001
```

### 4.3 访问
- 校内/SSH 转发：`http://10.108.17.241:5001`
- 若端口被占用，可改 `app.run(port=500x)`。

---

## 5. 功能与页面
- `/` 与 `/chat_system`：对话式数字人（Qwen + Edge-TTS + 渲染）。
- `/video_generation`：输入文本或上传音频生成视频。
- `/model_training`：上传视频启动训练，查看进度与日志。
- `/inference`：上传音频，选择模型进行推理。

核心 API 见 `app.py` 中的路由：上传视频/音频、配置保存、模型列表、训练任务管理、推理生成等。

---

## 6. 渲染与内存优化要点（来自 RENDER_LOG）
- `render.py` 改为**增量写视频**，避免一次性拼接导致内存溢出；必要时使用分段渲染脚本 `run_render_chunks.ps1`。
- 自定义音频时，会自动扩展视频帧数以匹配音频长度，避免早停。
- 支持 `start_frame/end_frame` 子区间渲染以控制显存/内存占用。

---

## 7. 训练与依赖兼容性（来自 TRAINING_LOG）
- 如需训练，建议显存 ≥24GB。
- 针对 `open3d`/`pytorch3d` 冲突，可分环境执行（示例：`pytorch3d_safe` 与 `open3d_env`）。
- 兼容性补丁：
  - `data_utils/face_tracking/face_tracker.py` 增加 NumPy 2.0 兼容加载。
  - `utils/point_utils.py` 对 `open3d` 可选导入。
  - `train.py` 兼容 `mmcv/mmengine` 的 Config 加载。

---

## 8. 数据与模型准备（简要）
- 示例数据可参考 `data/Obama`；若自定义数据，放入 `data/<name>/` 并确保视频分辨率适配。
- 面部解析权重：`data_utils/face_parsing/79999_iter.pth`。
- 若使用 DeepSpeech 特征，需在 `egstalker_tf_py39` 环境内运行对应提取脚本（见 `run_render_chunks.ps1` 或 `EGSTalker_Model/inference.py` 中的路径配置）。

---

## 9. 服务器操作速查
- 连接：`ssh -p 1986 xsj@10.108.17.241`
- 上传项目：`scp -P 1986 -r "<local_TFG_ui_path>" xsj@10.108.17.241:~/projects/egstalker/`
- 监控：`nvtop`、`htop`、`nvidia-smi`
- 端口占用：`netstat -tuln | grep 5001`
- 清理：`du -sh *` 查大文件，控制 <200GB。

---

## 10. 常见问题
- **无法访问页面**：确认渲染后端与主 UI 均已启动；端口未被防火墙阻挡；使用 HTTP 而非 HTTPS。
- **端口被占用**：更换 5001–5100；或终止占用进程。
- **CUDA OOM/显存不足**：减小 batch，分段渲染，或使用子区间渲染参数。
- **GPU 温度高**：降低并发，等待降温，避免占用 GPU6/7。
- **依赖冲突**：按建议的多环境方案分开安装；遇到 `open3d`/`pytorch3d` 冲突时分环境运行。

---