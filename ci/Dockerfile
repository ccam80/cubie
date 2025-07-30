# CUDA 12.9 on Ubuntu 24.04 (pin a specific patch for reproducibility)
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps: Python 3.12, pip, git, build tools (helpful for wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    git build-essential ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
 && python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Set "python" to 3.12 and upgrade pip toolchain
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
COPY ci/requirements.ci.txt /tmp/requirements.ci.txt
RUN python -m pip install -r /tmp/requirements.ci.txt

# Working directory for the job

# Default shell
CMD ["bash"]