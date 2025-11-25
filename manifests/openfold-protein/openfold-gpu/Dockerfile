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
# Install PyTorch + CUDA 11.8
# (adjust versions if you prefer different torch/torchvision/torchaudio)
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
# IMPORTANT: disable build isolation so torch is visible
RUN pip install --no-cache-dir --no-build-isolation -e .

# ------------------------------------------------------------------
# Bring your scripts into the image
# ------------------------------------------------------------------
WORKDIR /app
COPY run_pretrained_openfold.py .
COPY run_gpu_inference.py .

# Optional: set defaults for paths (you'll likely still override via env/args)
# ENV FASTA_ROOT=/data/input_fasta_single \
#     OUTPUT_ROOT=/data/predictions \
#     PRECOMPUTED_ALIGNMENT_DIR=/data/databases/embeddings_output_dir \
#     MMCIF_DIR=/data/databases/pdb_mmcif/mmcif_files \
#     OPENFOLD_CHECKPOINT_PATH=/data/openfold_params/finetuning_ptm_1.pt

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

ENTRYPOINT ["python", "run_gpu_inference.py"]