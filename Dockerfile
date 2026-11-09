# Imagen de la API de MiBici con el modelo P2 del Model Registry ya incluido.
# El modelo viene versionado en models/serving/p2; se actualiza con make recover-model.
FROM python:3.12.10-slim

# MIBICI_MODEL_SOURCE=bundled hace que la API cargue la copia empaquetada y no el registry
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    MIBICI_MODEL_SOURCE=bundled

WORKDIR /app

# Primero las dependencias, para que Docker reutilice esta capa si solo cambia el código.
# La línea "-e ." se quita porque el código se usa directo desde /app (PYTHONPATH).
# Si un antivirus inspecciona HTTPS (por ejemplo Norton), su certificado raíz se pasa como
# secreto opcional con --secret id=ca_extra,src=<archivo.pem>; no queda guardado en la imagen.
COPY requirements.txt .
RUN --mount=type=secret,id=ca_extra,required=false \
    grep -v "^-e" requirements.txt > /tmp/dependencias.txt \
    && if [ -f /run/secrets/ca_extra ]; then \
        cat /etc/ssl/certs/ca-certificates.crt /run/secrets/ca_extra > /tmp/ca.pem \
        && export PIP_CERT=/tmp/ca.pem; \
    fi \
    && pip install -r /tmp/dependencias.txt \
    && rm -f /tmp/dependencias.txt /tmp/ca.pem

# Código del proyecto y modelo descargado del registry
COPY mibici_ml ./mibici_ml
COPY models/serving ./models/serving

# La API corre con un usuario sin privilegios
RUN useradd --create-home --uid 1000 mibici
USER mibici

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "mibici_ml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
