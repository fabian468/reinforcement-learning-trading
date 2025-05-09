# Usamos una imagen base de Python
FROM python:3.10-slim


# Establecemos el directorio de trabajo en el contenedor
WORKDIR /app

# Copiamos el archivo de requisitos
COPY requirements.txt .

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el contenido del proyecto
COPY . .

# Expone el puerto si es necesario (por ejemplo, para Jupyter o app)
EXPOSE 8888

# Comando para ejecutar el script de Python
CMD ["python", "run.py"]
