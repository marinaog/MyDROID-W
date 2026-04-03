# Use NVIDIA CUDA 11.8 base image with development tools
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
# Set CUDA architecture list for compatibility (RTX 30xx/40xx/A-series)
ENV TORCH_CUDA_ARCH_LIST="7.5"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    bzip2 \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm miniconda.sh

# Initialize conda for shell interaction
RUN conda init bash

# Set working directory
WORKDIR /app

# --- Step 1: Aceptar ToS y Crear Entorno Base ---
# Resolvemos el error de "Terms of Service" antes de crear el env
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n droid-w python=3.10 -y

# --- Step 2: Install CUDA Toolkit y PyTorch ---
# Separamos esto para que la caché de Docker guarde las descargas pesadas
RUN conda run -n droid-w conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit && \
    conda run -n droid-w pip install numpy==1.26.3 && \
    conda run -n droid-w pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# --- Step 3: Install Specialized Torch Extensions ---
RUN conda run -n droid-w pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html && \
    conda run -n droid-w pip install -U xformers==0.0.22.post7+cu118 --index-url https://download.pytorch.org/whl/cu118

# --- Step 4: Install Submodules (Thirdparty) ---
COPY thirdparty/ /app/thirdparty/

# Forzamos variables de entorno para una compilación limpia
ENV MAX_JOBS=2
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1

# Instalamos los submódulos uno a uno, entrando en sus carpetas
RUN conda run -n droid-w pip install --upgrade setuptools==69.5.1 ninja

# Instalación manual de lietorch (el que más problemas da)
WORKDIR /app/thirdparty/lietorch
RUN conda run -n droid-w python setup.py install

# Instalación de los demás
WORKDIR /app/thirdparty/diff-gaussian-rasterization-w-pose
RUN conda run -n droid-w python setup.py install

WORKDIR /app/thirdparty/simple-knn
RUN conda run -n droid-w python setup.py install

# Volvemos al directorio principal
WORKDIR /app


# --- Step 5: Install Main Package and Requirements ---
COPY . .
RUN conda run -n droid-w python -m pip install -e . --no-build-isolation && \
    conda run -n droid-w python -m pip install -r requirements.txt

# --- Step 6: Install MMCV ---
RUN conda run -n droid-w pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

# Setup shell to activate environment on entry
RUN echo "conda activate droid-w" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Final check of the installation
RUN conda run --no-capture-output -n droid-w python -c "import torch; import lietorch; import simple_knn; import diff_gaussian_rasterization; print('CUDA Available:', torch.cuda.is_available())"

CMD ["bash"]
