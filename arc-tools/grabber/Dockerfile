FROM python:3.12-slim

WORKDIR /app

# System deps for psycopg2-binary, lxml, curl_cffi
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY vendor/ vendor/
COPY data/ data/
COPY temporal_worker.py temporal_activities.py temporal_workflows.py temporal_workflows_v2.py temporal_activities_v2.py ./

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
