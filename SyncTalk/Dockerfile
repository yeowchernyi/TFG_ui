FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 设置环境变量
ENV PYTHON_VERSION=3.8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    PYTHONPATH=/SyncTalk

# 安装系统依赖
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    gcc \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-venv \
    python3-tk \
    tk-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    wget \
    curl \
    unzip \
    portaudio19-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && python3 -m pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 设置工作目录
WORKDIR /SyncTalk

# 复制项目文件
COPY . .

# 克隆预训练模型，避免每次启动容器重新下载
COPY ./.cache/torch/hub/checkpoints /root/.cache/torch/hub/checkpoints

# 安装所有Python依赖
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install -r requirements.txt && \
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html && \
    pip install tensorflow-gpu==2.8.1 && \
    pip install ./freqencoder ./shencoder ./gridencoder ./raymarching && \
    rm -rf ./.cache/ && \
    pip cache purge

# # 下载构建环境太慢，改为使用预下载的本地包安装，构建前需在本地执行以下命令
# # pip download torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 -d ./packages
# # pip download -r requirements.txt -d ./packages
# RUN pip install --no-index --find-links ./packages torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 && \
#     pip install --no-index --find-links ./packages -r requirements.txt && \
#     pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html && \
#     pip install --no-index --find-links ./packages tensorflow-gpu==2.8.1 && \
#     pip install ./freqencoder ./shencoder ./gridencoder ./raymarching && \
#     rm -rf ./packages/ ./.cache/ && \
#     pip cache purge


# 默认命令
CMD ["python", "--help"]