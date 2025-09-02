# Model-Based RL Human Intent Recognition System
# Production Docker Container with Research-Grade Validation

FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src \
    ENVIRONMENT=production

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (skip problematic ones)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy scipy scikit-learn matplotlib seaborn pandas && \
    pip install --no-cache-dir torch tqdm joblib statsmodels plotly

# Copy source code
COPY src/ ./src/
COPY run_*.py ./
COPY *.md ./
COPY setup.py .

# Install project in development mode
RUN pip install --no-cache-dir -e . --no-deps

# Create directories for results
RUN mkdir -p /app/results /app/logs /app/monitoring

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash research && \
    chown -R research:research /app
USER research

# Expose port for potential web interfaces
EXPOSE 8080 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import src; print('✅ System healthy')" || exit 1

# Default command
CMD ["python3", "run_performance_benchmarks.py"]