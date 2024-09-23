FROM vllm/vllm-openai:v0.6.1.post2 

COPY --from=ghcr.io/astral-sh/uv:0.4.10 /uv /bin/uv

ADD src pyproject.toml uv.lock /app/

RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system --no-cache-dir "/app/[eval]"
