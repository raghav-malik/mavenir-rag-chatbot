FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download embedding and reranker models during build
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('farbodtavakkoli/OTel-Embedding-34M'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]