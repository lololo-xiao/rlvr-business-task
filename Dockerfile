FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Build the SQLite business data at image build time (deterministic seed)
RUN python -m rlvr_task.data.build_db
CMD ["python", "-m", "examples.run_examples"]
