FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev git && \
    ln -s /usr/bin/python3.12 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Crear entorno virtual (opcional pero recomendable)
RUN python3 -m venv /env
ENV PATH="/env/bin:$PATH"

# Instalar pip y dependencias compatibles
RUN pip install --upgrade pip

# PyTorch + CUDA 12.4 (nightly)
RUN pip install --pre torch==2.7.0.dev20250310+cu124 torchvision==0.22.0.dev20250310+cu124 torchaudio==2.7.0.dev20250310+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124

# Resto de requerimientos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Puerto si usas Jupyter u otros servicios
EXPOSE 8888

# Comando por defecto
CMD ["python3", "run_con_per.py"]
