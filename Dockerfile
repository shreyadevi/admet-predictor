# ADMET prediction API. Models are trained during the build, so the image is
# self-contained and reproducible from the committed data.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# RDKit's drawing code links against these libraries.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api
COPY data ./data
RUN pip install -e ".[api]"

# Pass --build-arg TRAIN_ARGS=--fast for a quicker build with smaller models.
ARG TRAIN_ARGS=""
RUN admet train ${TRAIN_ARGS} --out /app/reports

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
# Cloud Run sets $PORT; default to 8080 elsewhere.
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
