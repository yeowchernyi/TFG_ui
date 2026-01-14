#  EGSTalker UI: 交互式数字人生成系统

这是一个集成了 **Qwen 大语言模型** 与 **EGSTalker 高保真数字人渲染** 的全栈演示系统。用户可以通过网页端与 AI 对话，并实时生成对应的数字人驱动视频。

---

## 🛠️ 环境准备 (Environment Setup)

本项目涉及多个深度学习框架，建议创建三个独立的 Conda 环境以避免依赖冲突：

1.  **主环境 (`egstalker_py39`)**: 用于运行主 UI、LLM 推理及大部分预处理步骤。
2.  **追踪环境 (`egstalker_track_py39`)**: 专门用于步骤 2 的面部追踪（3D Tracking）。
3.  **TF 环境 (`egstalker_tf_py39`)**: 专门用于步骤 8-9 的 DeepSpeech 特征提取。

---

## 🚀 快速启动 (Quick Start)

要完整运行系统，需要**同时开启**两个终端窗口，分别启动主后端和渲染后端。

### 1. 启动主控后端 (Main Backend & UI)
负责网页展示、逻辑调度及 Qwen 模型对话。
```bash
conda activate egstalker_py39
cd TFG_ui
python app.py
```

### 2. 启动渲染后端 (Rendering Service)
负责接收音频并生成视频（默认端口 5002）。
```bash
conda activate egstalker_py39
cd TFG_ui/EGSTalker_Model
python app.py
```

---

## 🔄 预处理流程与环境切换 (Preprocessing Pipeline)

如果你需要处理新的素材（Source Data），请严格按照以下步骤及对应的环境进行操作：

| 步骤 | 任务描述 | 所需 Conda 环境 | 配置文件 |
| :--- | :--- | :--- | :--- |
| **Step 1** | 数据初始化 | `egstalker_py39` | `egstalker_py39.yml` |
| **Step 2** | **面部追踪 (3D Tracking)** | `egstalker_track_py39` | `egstalker_track_py39.yml` |
| **Step 3-7** | 特征对齐与处理 | `egstalker_py39` | `egstalker_py39.yml` |
| **Step 8-9** | **DeepSpeech 特征提取** | `egstalker_tf_py39` | `egstalker_tf_py39.yml` |

---

## ⚠️ 关键配置 (Critical Configuration)

### 1. 修改 TensorFlow 环境路径
由于渲染后端需要调用特定的 TensorFlow 环境进行音频特征提取，请务必修改以下文件中的路径：

**文件位置**: `TFG_ui/EGSTalker_Model/inference.py`
```python
# 找到这一行，修改为你本地 egstalker_tf_py39 环境的 python 绝对路径
TF_PYTHON = "/home/your_user/.conda/envs/egstalker_tf_py39/bin/python"
```

### 2. GPU 自动分配逻辑
系统内置了智能 GPU 调度逻辑：
- 自动检测并选择显存最空闲的 GPU。（默认自动避开 GPU 6 和 GPU 7）
- 渲染后端与 LLM 后端会分别独立寻找最空闲卡，实现负载均衡。

---

## 📂 项目结构

- `backend/`: Qwen 大模型驱动引擎。
- `EGSTalker_Model/`: 数字人推理核心模块及 Flask 渲染服务。
- `data/`: 存放预处理素材（如 Obama 示例数据）。
- `output/`: 渲染生成的视频结果。
- `static/` & `templates/`: 前端网页资源。

---

## 💡 备注
- **模型训练**: 本仓库目前主要用于推理展示。如需训练新人物，请确保显存 > 24GB 并参考各模块下的 `train.py`。
- **路径问题**: 建议所有操作在项目根目录 `TFG_ui` 下进行，以保证相对路径引用正确。

---

## 致谢
- 本项目包含了来自EGSTalker的代码。我们已经修改了render.py以支持我们的 UI 集成。
