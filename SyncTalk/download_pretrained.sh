#!/bin/bash

set -e

echo "开始下载 SyncTalk 所需模型文件..."

# 创建目录
mkdir -p data_utils/face_parsing
mkdir -p data_utils/face_tracking/3DMM
mkdir -p .cache/torch/hub/checkpoints
mkdir -p .cache/torch/hub/checkpoints/lpips

# 下载函数（带重试）
download_with_retry() {
    local url=$1
    local output=$2
    local retries=3
    
    for i in $(seq 1 $retries); do
        echo "下载: $(basename $output) (尝试 $i/$retries)..."
        
        # 如果文件已存在且大小合理，跳过
        if [ -f "$output" ] && [ -s "$output" ]; then
            local size=$(du -h "$output" | cut -f1)
            echo "✓ 已存在: $(basename $output) ($size)"
            return 0
        fi
        
        # 尝试 wget
        if wget -q --show-progress "$url" -O "$output.tmp" 2>/dev/null; then
            mv "$output.tmp" "$output"
            local size=$(du -h "$output" | cut -f1)
            echo "✓ 下载成功: $(basename $output) ($size)"
            return 0
        fi
        
        # 尝试 curl
        if curl -L --progress-bar "$url" -o "$output.tmp" 2>/dev/null; then
            mv "$output.tmp" "$output"
            local size=$(du -h "$output" | cut -f1)
            echo "✓ 下载成功: $(basename $output) ($size)"
            return 0
        fi
        
        rm -f "$output.tmp"
        echo "下载失败，等待 5 秒后重试..."
        sleep 5
    done
    
    echo "最终下载失败: $(basename $output)"
    return 1
}

# 下载列表
declare -A downloads=(
    # 人脸解析模型
    ["https://github.com/YudongGuo/AD-NeRF/raw/master/data_util/face_parsing/79999_iter.pth"]="data_utils/face_parsing/79999_iter.pth"
    
    # 3DMM 模型
    ["https://github.com/YudongGuo/AD-NeRF/raw/master/data_util/face_tracking/3DMM/exp_info.npy"]="data_utils/face_tracking/3DMM/exp_info.npy"
    ["https://github.com/YudongGuo/AD-NeRF/raw/master/data_util/face_tracking/3DMM/keys_info.npy"]="data_utils/face_tracking/3DMM/keys_info.npy"
    ["https://github.com/YudongGuo/AD-NeRF/raw/master/data_util/face_tracking/3DMM/sub_mesh.obj"]="data_utils/face_tracking/3DMM/sub_mesh.obj"
    ["https://github.com/YudongGuo/AD-NeRF/raw/master/data_util/face_tracking/3DMM/topology_info.npy"]="data_utils/face_tracking/3DMM/topology_info.npy"
    
    # PyTorch 预训练模型
    ["https://download.pytorch.org/models/resnet18-5c106cde.pth"]=".cache/torch/hub/checkpoints/resnet18-5c106cde.pth"
    ["https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"]=".cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
    ["https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth"]=".cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth"
    ["https://download.pytorch.org/models/vgg16-397923af.pth"]=".cache/torch/hub/checkpoints/vgg16-397923af.pth"
    
    # 人脸对齐模型
    ["https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"]=".cache/torch/hub/checkpoints/s3fd-619a316812.pth"
    ["https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip"]=".cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip"
    
    # LPIPS 模型
    ["https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"]=".cache/torch/hub/checkpoints/lpips/alex.pth"
)

# 执行下载
success_count=0
total_count=${#downloads[@]}

for url in "${!downloads[@]}"; do
    output="${downloads[$url]}"
    if download_with_retry "$url" "$output"; then
        ((success_count++))
    fi
done

echo ""
echo "下载统计: $success_count/$total_count 个文件成功"

if [ $success_count -eq $total_count ]; then
    echo "所有模型下载完成！"
else
    echo "部分文件下载失败"
fi

echo ""
echo "下一步操作:"
echo "1. 手动下载 Basel Face Model:"
echo "   访问: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads"
echo "   下载后移动 01_MorphableModel.mat 到 data_utils/face_tracking/3DMM/"
echo ""
echo "2. 转换 BFM 模型:"
echo "   cd data_utils/face_tracking && python convert_BFM.py"

# 如果 BFM 文件存在，自动转换
if [ -f "data_utils/face_tracking/3DMM/01_MorphableModel.mat" ]; then
    echo ""
    echo "检测到 BFM 模型，正在转换..."
    cd data_utils/face_tracking && python convert_BFM.py && cd ../..
    echo "BFM 模型转换完成"
fi

echo ""
echo "所有任务完成！"