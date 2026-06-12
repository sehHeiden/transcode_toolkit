FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY config.yaml config.yaml

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["transcode-toolkit"]
