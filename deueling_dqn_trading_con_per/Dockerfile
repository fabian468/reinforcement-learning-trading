# Imagen base con soporte para CUDA 11.8 + Python
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Evita interacción durante instalaciones
ENV DEBIAN_FRONTEND=noninteractive

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y python3 python3-pip git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar todo el código fuente
COPY . .

# Exponer puerto para Jupyter o visualización si es necesario
EXPOSE 8888

# Comando por defecto: ejecuta tu agente RL
CMD ["python3", "run_con_per.py"]
