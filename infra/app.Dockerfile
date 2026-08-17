# Streamlit app image. Build context is the repo root:
#   docker compose build app
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first, so code edits don't re-resolve the environment.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Only what the app imports at runtime.
COPY services/ services/
COPY agent/ agent/
COPY app/ app/

# Bake the ONNX embedding model (~90 MB) into the image so the container
# needs no network access at query time.
RUN uv run python -m services.embedder

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "app/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
