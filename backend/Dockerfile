# ── Stage 1: Builder ──────────────────────────────────────────────────────
# Build context: project root  →  docker build -f backend/Dockerfile .
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy backend source
COPY backend/src/     ./backend/src/
COPY backend/main.py  ./backend/main.py
COPY backend/.env.example ./backend/.env

# Copy frontend assets (Flask reads from ../frontend relative to backend/)
COPY frontend/static/    ./frontend/static/
COPY frontend/templates/ ./frontend/templates/

# Expose port
EXPOSE 5000

# Environment defaults (can be overridden at docker run)
ENV MARKET_DATA_PROVIDER=yfinance \
    MARKET_DATA_API_KEY= \
    PORT=5000 \
    DEBUG=False \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/')" || exit 1

# Run from backend/ so relative imports work
WORKDIR /app/backend
CMD ["python", "main.py"]
