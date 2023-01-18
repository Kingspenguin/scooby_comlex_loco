FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04
WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake \
        curl \
        gcc-8 \
        g++-8 \
        wget \
        bzip2 \
        git \
        vim \
        tmux \
        git \
        unzip \
        build-essential \
        vulkan-utils \
        mesa-common-dev \
        mesa-vulkan-drivers \
        libosmesa6-dev \
        libgl1-mesa-glx \
        libglfw3 \
        patchelf \
        libglu1-mesa \
        libxext6 \
        graphviz \
        libxtst6 \
        libxrender1 \
        libxi6 \
        libegl1 \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        mpich \
        ninja-build

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

COPY libs/IsaacGym_Preview_4_Package/isaacgym /workspace/isaacgym


RUN apt-get update \
    && cd /workspace/isaacgym \
    && bash create_conda_env_rlgpu.sh \
    && conda clean -ya \
    && cd python \
    && /opt/conda/envs/rlgpu/bin/pip install -e . 

RUN /opt/conda/envs/rlgpu/bin/pip install pytorch3d \
    ninja trimesh wandb six lxml matplotlib gym opencv-python \
    torch torchvision numpy gym

RUN rm -rf /root/.cache/pip
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /opt/conda/envs/rlgpu/lib:$LD_LIBRARY_PATH

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
