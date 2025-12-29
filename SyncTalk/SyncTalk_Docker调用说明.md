# SyncTalk Docker 调用说明

## 构建Docker镜像
```bash
git clone https://github.com/ZiqiaoPeng/SyncTalk.git
cd SyncTalk
# 将Dockerfile, .dockerignore, download_pretrained.sh移动到SyncTalk
./download_pretrained.sh
# 按指引下载01_MorphableModel.mat
docker build -t synctalk .
```

## 打包Docker镜像
```bash
docker save -o synctalk.tar synctalk
```

## 从tar导入Docker镜像
```bash
docker load -i synctalk.tar
```

## 脚本文件

`run_synctalk.sh`

## 目录结构

```
当前工作目录/
└── SyncTalk/                    # 自动创建的工作目录
    ├── data/                    # 数据目录
    │   └── {视频名称}/          # 预处理数据文件夹
    ├── model/                   # 模型目录
    │   └── {视频名称}_ep{N}/    # 训练好的模型
    │       ├── checkpoints/     # 模型检查点
    │       └── results/         # 推理结果
    └── audio/                   # 音频文件目录
```

## 脚本调用命令

### train - 完整训练流程
```bash
./run_synctalk.sh train \
    --video_path <视频文件路径> \
    --gpu <GPU设备> \
    --epochs <训练轮数>
```

**参数说明：**
- `--video_path`: 输入视频文件路径（必需）
- `--gpu`: GPU设备（默认: GPU0）
  - `GPU0`, `GPU1`, `GPU2`, ... - 指定GPU
  - `CPU` - 使用CPU
- `--epochs`: 训练轮数（默认: 140）

**示例：**
```bash
./run_synctalk.sh train --video_path ./my_video.mp4 --gpu GPU1 --epochs 100
```

### preprocess - 仅数据预处理
```bash
./run_synctalk.sh preprocess_only \
    --video_path <视频文件路径> \
    --gpu <GPU设备>
```

**示例：**
```bash
./run_synctalk.sh preprocess_only --video_path ./training_video.mp4 --gpu GPU0
```

### train_only - 仅模型训练
```bash
./run_synctalk.sh train_only \
    --video_path <视频文件路径> \
    --gpu <GPU设备> \
    --epochs <训练轮数>
```

**注意：** 需要先运行预处理步骤生成数据

**示例：**
```bash
./run_synctalk.sh train_only --video_path ./training_video.mp4 --gpu GPU0 --epochs 80
```

### infer - 视频推理
```bash
./run_synctalk.sh infer \
    --model_dir <模型目录路径> \
    --audio_path <音频文件路径> \
    --gpu <GPU设备>
```

**参数说明：**
- `--model_dir`: 模型目录路径（模型名称格式: `视频名称_ep轮数`）
- `--audio_path`: 驱动音频文件路径
- `--gpu`: GPU设备

**示例：**
```bash
./run_synctalk.sh infer \
    --model_dir ./SyncTalk/model/my_video_ep50 \
    --audio_path ./speech.wav \
    --gpu GPU0
```

## 输出文件说明

### 训练完成后
- **数据目录**: `SyncTalk/data/{视频名称}/`
- **模型目录**: `SyncTalk/model/{视频名称}_ep{轮数}/`

### 推理完成后
- **音频目录**: `SyncTalk/audio/{音频}`
- **输出视频**: `SyncTalk/model/{视频名称}_ep{轮数}/results/{视频名称}_ep{轮数}_{音频名称}.mp4`

## 测试模式
在不构建 Docker 镜像的情况下测试脚本逻辑：

```bash
# 启用测试模式
TEST_MODE=1 ./run_synctalk.sh train --video_path ./test.mp4 --gpu GPU0 --epochs 10
TEST_MODE=1 ./run_synctalk.sh infer --model_dir test_ep10 --audio_path ./test.wav
```

**示例：**
```
SyncTalk/data/my_video
SyncTalk/model/my_video_ep50
SyncTalk/audio/speech.wav
SyncTalk/model/my_video_ep50/results/my_video_ep50_speech.mp4
```

## 注意事项

1. **首次使用**需要先构建 Docker 镜像
2. **视频文件**应为常见的视频格式（mp4, avi, mov等）
3. **音频文件**应为常见的音频格式（wav, mp3等）
