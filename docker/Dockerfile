ARG BASE_IMAGE=trtllm:1.3.0rc20
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    GRADIO_ANALYTICS_ENABLED=false \
    TOKENIZERS_PARALLELISM=false

WORKDIR /opt/dots-mocr

COPY pyproject.toml LICENSE NOTICE ./
COPY src ./src
COPY demo ./demo
COPY docker ./docker
COPY scripts ./scripts
COPY tests ./tests
COPY reports ./reports

RUN python3 -m pip install --no-cache-dir --no-build-isolation . \
    && python3 -c "import transformers; assert transformers.__version__ == '5.5.4'" \
    && python3 -c "import fastapi, uvicorn, PIL; print('demo deps ok')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTHONPATH=/opt/dots-mocr

WORKDIR /workspace

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=3m --retries=3 \
    CMD python3 -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('DOTS_MOCR_WEB_PORT', '7860') + '/healthz', timeout=3).read()" || exit 1

CMD ["python3", "-m", "demo.server", "--help"]
