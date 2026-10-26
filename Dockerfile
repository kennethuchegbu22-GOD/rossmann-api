FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Render sets $PORT (default 10000). One worker keeps memory low on the free tier.
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 120