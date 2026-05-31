FROM python:3.12-slim

# Set environment variables for Hugging Face cache and Transformers
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements file and install dependencies
COPY image_embedding_pipeline/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# Copy the entire image_embedding_pipeline directory structure
COPY image_embedding_pipeline/config ./config/
COPY image_embedding_pipeline/src ./src/

# Create logs directory for logging output
RUN mkdir -p /app/logs

# Health check - optional, can be removed if not needed
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command to run the pipeline in test mode
# Can be overridden with: docker run ... python src/run.py prod
CMD ["python", "src/run.py", "test"]

# Uncomment the line below to run in PRODUCTION mode instead of test
# CMD ["python", "src/run.py", "prod"]