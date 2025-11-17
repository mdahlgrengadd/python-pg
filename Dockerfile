# Multi-stage Dockerfile for WebWork Python

# Stage 1: Base image with Python dependencies
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY webwork_api/requirements_v2.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development image
FROM base as development

# Install development tools
RUN pip install --no-cache-dir \
    black \
    ruff \
    mypy \
    pytest \
    pytest-cov \
    pytest-asyncio

# Copy application code
COPY . /app/

# Install pg package in editable mode
RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "webwork_api.main_v2:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: Production image
FROM base as production

# Copy only necessary files
COPY pg/ /app/pg/
COPY webwork_api/ /app/webwork_api/
COPY setup.py /app/
COPY README.md /app/

# Install pg package
RUN pip install .

# Create non-root user
RUN useradd -m -u 1000 webwork && \
    chown -R webwork:webwork /app

USER webwork

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["uvicorn", "webwork_api.main_v2:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
