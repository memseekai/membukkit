# syntax=docker/dockerfile:1
FROM python:3.12-slim

# CPU-only torch keeps the image ~2GB instead of ~7GB with CUDA libs.
# MEMBUKKIT_HOME=/data puts stores, model weights, and saved keys in one volume.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    PYTHONUNBUFFERED=1 \
    MEMBUKKIT_HOME=/data

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip pip install ".[all]"

RUN useradd -m -u 1000 membukkit && mkdir -p /data && chown membukkit /data
USER membukkit
VOLUME /data
EXPOSE 8377

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8377/api/health')"

CMD ["membukkit", "ui", "--host", "0.0.0.0", "--no-browser"]
