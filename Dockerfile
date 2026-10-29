# Етап 1: Builder з використанням uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/

RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install . && \
    findex index data/ -o /app/index.bin

# Етап 2: Фінальний мінімальний образ без root-прав
FROM python:3.12-slim-bookworm AS runtime

RUN groupadd -r findex && useradd -r -g findex -m findex

WORKDIR /app

COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=builder --chown=findex:findex /app/src /app/src
COPY --from=builder --chown=findex:findex /app/data /app/data
COPY --from=builder --chown=findex:findex /app/index.bin /app/index.bin

ENV PATH="/app/.venv/bin:$PATH"
ENV INDEX_PATH="/app/index.bin"
ENV CORPUS_DIR="/app/data"
ENV PORT=8000

USER findex

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["findex", "serve", "--host", "0.0.0.0", "--port", "8000"]
