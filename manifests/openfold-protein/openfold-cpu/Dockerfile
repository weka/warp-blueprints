FROM condaforge/mambaforge:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Avoid buildx warning about undefined vars by using default expansion
ENV LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
    LIBRARY_PATH=/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}

# --- OS deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

SHELL ["bash", "-c"]

# ------------------------------------------------------------------
# Configure conda channels explicitly (conda-forge + bioconda)
# ------------------------------------------------------------------
RUN conda config --system --add channels conda-forge && \
    conda config --system --add channels bioconda && \
    conda config --system --set channel_priority strict

# ------------------------------------------------------------------
# Create env with alignment tools from bioconda
# hmmer   -> jackhmmer
# hhsuite -> hhsearch
# kalign3 -> kalign (or use kalign2 if kalign3 not available)
# mmseqs2 -> MMseqs2
# ------------------------------------------------------------------
RUN mamba create -y -n cpu-env python=3.10 \
        hmmer \
        hhsuite \
        kalign3 \
        mmseqs2 \
        pip

# Make the env active for subsequent RUNs
RUN echo "conda activate cpu-env" >> ~/.bashrc
ENV PATH="/opt/conda/envs/cpu-env/bin:${PATH}"

# --- Python deps ---
COPY requirements_cpu.txt /tmp/requirements_cpu.txt
RUN pip install --no-cache-dir -r /tmp/requirements_cpu.txt

# --- App code ---
WORKDIR /app
COPY run_cpu_pipeline.py .

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

ENTRYPOINT ["python", "run_cpu_pipeline.py"]