# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas numpy ccxt


# Production stage
FROM python:3.12-slim AS production

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash trader
USER trader

# Copy application code
COPY --chown=trader:trader src/ ./src/
COPY --chown=trader:trader config/ ./config/
COPY --chown=trader:trader main.py .
COPY --chown=trader:trader pyproject.toml .

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Default command
CMD ["python", "main.py"]


# Development stage
FROM production AS development

USER root

# Install dev dependencies
RUN /opt/venv/bin/pip install --no-cache-dir pytest pytest-cov ruff mypy

# Copy test files
COPY --chown=trader:trader tests/ ./tests/

USER trader

# Run tests by default in dev mode
CMD ["pytest", "tests/", "-v"]


# Test stage (for CI)
FROM development AS test

CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=term-missing"]
