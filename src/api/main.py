import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

load_dotenv()
logging.basicConfig(level="INFO")

app = FastAPI(
    title="CredAgent",
    description="Agentic Credit Decisioning & Risk Explainability System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "CredAgent",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
