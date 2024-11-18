# Use the NVIDIA CUDA base image
#FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
# Use the official NVIDIA CUDA base image with Ubuntu 22.04
FROM mirekphd/cuda-11.2-cudnn8-devel-ubuntu22.04:latest

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda clean -afy
ENV PATH=/opt/conda/bin:$PATH

# Create and activate the Conda environment
COPY MV2Denv.yml /tmp/MV2Denv.yml
RUN conda env create -f /tmp/MV2Denv.yml

# Activate the environment
SHELL ["conda", "run", "-n", "MV2Denv", "/bin/bash", "-c"]

#RUN pip3 install --upgrade pip
# Install PyTorch in the environment
#RUN conda install pytorch==1.11.0 torchvision torchaudio cudatoolkit=11.2 -c pytorch
# Install PyTorch 1.11 and its dependencies for CUDA 11.2
#RUN conda env update --file /tmp/MV2Denv.yml --prune
