ARG BASE_IMAGE=trtllm:1.3.0rc20
FROM ${BASE_IMAGE}

WORKDIR /opt/dots-mocr

COPY pyproject.toml LICENSE NOTICE ./
COPY src ./src

RUN python3 -m pip install --no-cache-dir --no-build-isolation . && python3 -c "import transformers; assert transformers.__version__ == '5.5.4'"

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

WORKDIR /workspace
