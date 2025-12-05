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
        libxml2 \
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

# Helpful for interactive shells (if you ever exec into the container)
RUN echo 'export PATH=/opt/conda/envs/gpu-env/bin:$PATH' >> ~/.bashrc || true

# ------------------------------------------------------------------
# Install full CUDA 11.8 toolkit (runtime + headers + nvcc)
# ------------------------------------------------------------------
RUN mamba install -y -n gpu-env -c "nvidia/label/cuda-11.8.0" \
        "cuda-toolkit=11.8.*" \
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

# ------------------------------------------------------------------
# Install OpenMM (GPU-enabled) into gpu-env
# ------------------------------------------------------------------
RUN mamba install -y -n gpu-env -c conda-forge \
        openmm \
        cudatoolkit=11.8 \
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
# ------------------------------------------------------------------
COPY requirements_gpu.txt /tmp/requirements_gpu.txt
RUN $PIP install --no-cache-dir -r /tmp/requirements_gpu.txt

# ------------------------------------------------------------------
# Clone OpenFold (source-only; we will NOT pip-install it as a package)
# ------------------------------------------------------------------
WORKDIR /app
RUN git clone https://github.com/aqlaboratory/openfold.git ./openfold

WORKDIR /app/openfold

# ------------------------------------------------------------------
# Download stereo_chemical_props.txt for Amber relaxation
# ------------------------------------------------------------------
RUN mkdir -p openfold/resources && \
    wget --no-check-certificate -O openfold/resources/stereo_chemical_props.txt \
      https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt && \
    touch openfold/resources/__init__.py

# ------------------------------------------------------------------
# Patch TensorRT import to be optional
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# (Optional) Patch attention_core arch list to avoid compute_89/sm_89
# ------------------------------------------------------------------
RUN $PYTHON - << 'PY'
from pathlib import Path

p = Path("openfold/utils/kernel/attention_core.py")
if p.exists():
    text = p.read_text()
    if "compute_89" in text or "sm_89" in text:
        text = text.replace("compute_89", "compute_86")
        text = text.replace("sm_89", "sm_86")
    if "89" in text:
        text = text.replace("89,", "")
        text = text.replace(", 89", "")
    p.write_text(text)
PY

# ------------------------------------------------------------------
# Build the fused CUDA extension attn_core_inplace_cuda manually
#   so attention_core.py can import it.
# ------------------------------------------------------------------
RUN cat > /app/openfold/setup_attn_core.py << 'EOF'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name="attn_core_inplace_cuda",
        sources=[
            "openfold/utils/kernel/csrc/softmax_cuda.cpp",
            "openfold/utils/kernel/csrc/softmax_cuda_kernel.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-maxrregcount=50",
                # Target safe architectures for CUDA 11.x
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_86,code=sm_86",
            ],
        },
    )
]

setup(
    name="attn_core_inplace_cuda",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
EOF

# Actually compile the extension in-place
RUN $PYTHON setup_attn_core.py build_ext --inplace

# ------------------------------------------------------------------
# We still do NOT pip install openfold as a package.
# We'll import it from source via PYTHONPATH.
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Bring your scripts into the image
# ------------------------------------------------------------------
WORKDIR /app
COPY run_pretrained_openfold.py .
COPY run_gpu_inference.py .

# Make both your app root and the OpenFold repo visible as top-level modules
ENV PYTHONPATH="/app:/app/openfold:${PYTHONPATH}"

# Force torch's own CUDA arch selection to a safe subset (runtime)
ENV TORCH_CUDA_ARCH_LIST="7.0 8.0 8.6"

# ------------------------------------------------------------------
# Non-root user
# ------------------------------------------------------------------
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# ------------------------------------------------------------------
# Entrypoint: always use gpu-env Python explicitly
# ------------------------------------------------------------------
ENTRYPOINT ["/opt/conda/envs/gpu-env/bin/python", "run_gpu_inference.py"]