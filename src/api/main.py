import logging
import os

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


@app.on_event("startup")
def _seed_on_startup():
    """Initialise the decision store and seed historical decisions if empty,
    so the monitoring dashboard has realistic data on a fresh deploy."""
    if os.getenv("CREDAGENT_AUTOSEED", "1") != "0":
        try:
            from src.seed_history import ensure_seeded
            n = ensure_seeded()
            if n:
                logging.getLogger(__name__).info("Seeded %d historical decisions.", n)
        except Exception as exc:  # pragma: no cover - non-fatal
            logging.getLogger(__name__).warning("Seed on startup skipped: %s", exc)


@app.get("/")
async def root():
    return {
        "service": "CredAgent",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
