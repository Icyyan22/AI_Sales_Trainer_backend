import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import auth, dashboard, feedback, messages, reports, scenarios, sessions
from app.models.db import async_session_factory, init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="AI Sales Trainer",
    description="AI Sales Training Chatbot - Multi-Agent System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": exc.__class__.__name__},
    )


app.include_router(auth.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(messages.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(scenarios.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
app.include_router(dashboard.router, prefix="/api/v1")


@app.get("/health")
async def health():
    from sqlalchemy import text

    db_status = "ok"
    try:
        async with async_session_factory() as db:
            await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"
    return {"status": "ok" if db_status == "ok" else "degraded", "database": db_status}
