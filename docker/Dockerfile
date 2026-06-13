FROM python:3.11-slim

WORKDIR /app

# System deps for xgboost / shap
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source, trained model artifacts and prebuilt vector store
COPY src/ src/
COPY models/ models/
COPY vectorstore/ vectorstore/
COPY streamlit_app.py .

EXPOSE 8000 8501

# Default: run the FastAPI service. Override the command to run Streamlit.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
