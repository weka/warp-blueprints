FROM condaforge/mambaforge:latest

# ------------------------------------------------------------------
# Basic env settings
# ------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/lib/x86_64-linux-gnu \
    LIBRARY_PATH=/lib/x86_64-linux-gnu

# ------------------------------------------------------------------
# OS deps
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        git \
        git-lfs \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# Use bash -lc so conda/mamba behave as expected
SHELL ["bash", "-lc"]

# ------------------------------------------------------------------
# Create minimal GPU env
# ------------------------------------------------------------------
RUN mamba create -y -n gpu-env python=3.10 pip && \
    conda clean -afy

# Make gpu-env *the* default Python for everything in this image
ENV PATH="/opt/conda/envs/gpu-env/bin:/opt/conda/bin:/opt/conda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# (Optional) For interactive shells inside the container
RUN echo 'export PATH=/opt/conda/envs/gpu-env/bin:$PATH' >> ~/.bashrc || true

# ------------------------------------------------------------------
# Install CUDA toolkit + nvcc into gpu-env
#   (needed so OpenFold can JIT-compile its custom CUDA kernels)
# ------------------------------------------------------------------
RUN mamba install -y -n gpu-env -c nvidia -c conda-forge \
        "cuda-toolkit=11.8" \
        "cuda-nvcc=11.8" \
    && conda clean -afy

# Point CUDA_HOME to the env and ensure nvcc / libs are visible
ENV CUDA_HOME=/opt/conda/envs/gpu-env
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# ------------------------------------------------------------------
# Install conda-level deps into gpu-env (before pip)
# ------------------------------------------------------------------
RUN mamba install -y -n gpu-env -c conda-forge \
        "pdbfixer>=1.9" \
        "numpy<2.0" \
    && conda clean -afy

# Use the env's pip & python explicitly
ENV PIP="/opt/conda/envs/gpu-env/bin/pip"
ENV PYTHON="/opt/conda/envs/gpu-env/bin/python"

# ------------------------------------------------------------------
# Install PyTorch + CUDA 11.8 into gpu-env
# ------------------------------------------------------------------
RUN $PIP install --no-cache-dir \
    torch==2.2.1 \
    torchvision==0.17.1 \
    torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu118

# ------------------------------------------------------------------
# Install OpenFold Python dependencies (excluding torch)
#   requirements_gpu.txt should NOT contain torch
#   but SHOULD contain things like:
#     modelcif
#     biopython
#     ml-collections
#     pandas
#     deepspeed
#     pytorch-lightning
# ------------------------------------------------------------------
COPY requirements_gpu.txt /tmp/requirements_gpu.txt
RUN $PIP install --no-cache-dir -r /tmp/requirements_gpu.txt

# ------------------------------------------------------------------
# Clone and install OpenFold in editable mode
# ------------------------------------------------------------------
WORKDIR /app
RUN git clone https://github.com/aqlaboratory/openfold.git ./openfold

WORKDIR /app/openfold

# Patch TensorRT import to be optional
RUN $PYTHON - << 'PY'
from pathlib import Path

p = Path("openfold/utils/script_utils.py")
text = p.read_text()

needle = "from .tensorrt_utils import instrument_with_trt_compile\n"
if needle in text:
    replacement = (
        "try:\n"
        "    from .tensorrt_utils import instrument_with_trt_compile\n"
        "except Exception:\n"
        "    # Fallback no-op when TensorRT / cuda-python are not installed\n"
        "    def instrument_with_trt_compile(model, *args, **kwargs):\n"
        "        return model\n"
    )
    text = text.replace(needle, replacement)
    p.write_text(text)
PY

# Install OpenFold itself (no build isolation so it can see torch & CUDA)
RUN $PIP install --no-cache-dir --no-build-isolation -e .

# ------------------------------------------------------------------
# Bring your scripts into the image
# ------------------------------------------------------------------
WORKDIR /app
COPY run_pretrained_openfold.py .
COPY run_gpu_inference.py .

# Make both your app root and the OpenFold repo visible as top-level modules
ENV PYTHONPATH="/app:/app/openfold:${PYTHONPATH}"

# ------------------------------------------------------------------
# Non-root user
# ------------------------------------------------------------------
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# ------------------------------------------------------------------
# Entrypoint: always use gpu-env Python
# ------------------------------------------------------------------
ENTRYPOINT ["python", "run_gpu_inference.py"]