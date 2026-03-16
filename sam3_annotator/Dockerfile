FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY server /app/server
COPY frontend /app/frontend

# Set working directory to server where main.py is
WORKDIR /app/server

# Expose FastAPI default port 8080 (or 8000 mapped to 8080)
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
