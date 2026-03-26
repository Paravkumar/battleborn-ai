FROM python:3.11-slim

WORKDIR /app

COPY src/ /app/src
COPY data/ /app/data

RUN pip install --no-cache-dir \
    "a2a-sdk[http-server]>=0.3.0" \
    click>=8.1.8 \
    httpx>=0.28.1 \
    openai>=1.57.0 \
    pydantic>=2.11.4 \
    python-dotenv>=1.1.0 \
    uvicorn>=0.34.2

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.__main__", "--host", "0.0.0.0", "--port", "5000"]

