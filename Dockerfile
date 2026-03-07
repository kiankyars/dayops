FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY dayops_core.py main.py backend.py ./

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
