FROM condaforge/mambaforge:latest

# Basic env settings
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/lib/x86_64-linux-gnu \
    LIBRARY_PATH=/lib/x86_64-linux-gnu

# --- OS deps ---
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

SHELL ["bash", "-c"]

# ------------------------------------------------------------------
# Create minimal GPU env
# ------------------------------------------------------------------
RUN mamba create -y -n gpu-env python=3.10 pip && \
    conda clean -afy

RUN echo "conda activate gpu-env" >> ~/.bashrc
ENV PATH="/opt/conda/envs/gpu-env/bin:${PATH}"

# ------------------------------------------------------------------
# Install conda-only dependencies BEFORE pip
# ------------------------------------------------------------------
RUN mamba install -y -n gpu-env -c conda-forge \
        "pdbfixer>=1.9" \
        "numpy<2.0" \
    && conda clean -afy

# ------------------------------------------------------------------
# Install PyTorch + CUDA 11.8
# ------------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.2.1 \
    torchvision==0.17.1 \
    torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu118

# ------------------------------------------------------------------
# Install OpenFold Python dependencies
# ------------------------------------------------------------------
COPY requirements_gpu.txt /tmp/requirements_gpu.txt
RUN pip install --no-cache-dir -r /tmp/requirements_gpu.txt

# ------------------------------------------------------------------
# Clone and install OpenFold in editable mode
# ------------------------------------------------------------------
WORKDIR /app
RUN git clone https://github.com/aqlaboratory/openfold.git ./openfold
WORKDIR /app/openfold

# Patch TensorRT import to be optional
RUN python - << 'PY'
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

RUN pip install --no-cache-dir --no-build-isolation -e .

# ------------------------------------------------------------------
# Bring your scripts into the image
# ------------------------------------------------------------------
WORKDIR /app
COPY run_pretrained_openfold.py .
COPY run_gpu_inference.py .

# Make both your app root and the OpenFold repo visible as top-level modules
ENV PYTHONPATH="/app:/app/openfold:${PYTHONPATH}"

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

ENTRYPOINT ["python", "run_gpu_inference.py"]