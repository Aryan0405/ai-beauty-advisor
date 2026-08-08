# Build context for this Dockerfile is the backend/ directory itself (see
# docker-compose.yml). Source is copied into /app/backend/... rather than
# /app/... so the "backend.app.*" absolute imports used throughout the
# codebase, and the PROJECT_ROOT paths several modules derive from
# __file__, resolve the same way inside the container as on a dev machine
# (PROJECT_ROOT walks up to /app, which is also where the data/ volume is
# mounted).
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

# Install dependencies first so this layer is cached across source changes.
# sentence-transformers pulls in torch; without pinning it to the CPU-only
# wheel index first, pip resolves the default PyPI build, which bundles the
# full CUDA toolkit and balloons the image from ~1.5GB to ~9GB for a
# workload that only ever runs CPU inference (faiss-cpu, no GPU anywhere).
COPY requirements.txt ./backend/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r backend/requirements.txt

COPY app ./backend/app
COPY ingestion ./backend/ingestion

# Run as a non-root user (spec section 17). The data/ mount point is
# created ahead of time so its ownership matches the app user; the actual
# SQLite DB and FAISS index are supplied at runtime via a volume mount, not
# baked into the image.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
