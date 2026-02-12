FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgl1 \
    ninja-build \
    pkg-config \
    wget \
 && rm -rf /var/lib/apt/lists/*

ENV PYTORCH=2.9.1 \
    CUDA=12.6 \
    CUDNN=9 \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    MMCV_WITH_OPS=1 \
    # Docker builds usually don't have GPU access, so PyTorch can't auto-detect
    # compute capability; set a reasonable default list to avoid an empty arch list.
    # Override at build time if you want a smaller binary.
    # Pascal GPUs (e.g., TITAN X (Pascal), Quadro P400) are compute capability 6.1.
    TORCH_CUDA_ARCH_LIST="6.1;6.1+PTX;7.5;8.0;8.6;8.9;9.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH=/opt/conda

RUN pip install -U openmim \
 && mim install mmengine \
 && mim install "mmcv<2.2.0,>=2.0.0rc4" \
 && mim install "mmdet<3.3.0,>=3.0.0" \
 && mim install "mmpose>=1.1.0"

RUN pip install --no-cache-dir distinctipy future tensorboard pyarrow h5py

# Install xtcocotools from source (to get around numpy binary incompatibility)
RUN pip uninstall -y xtcocotools || true \
 && pip install --no-binary=:all: "xtcocotools==1.14.3"
