from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text

from app.api.routes import auth, chat, exercises, programs, workouts
from app.core.config import settings
from app.core.rate_limit import limiter
from app.db.session import engine

app = FastAPI(title="AI Gym Coach API")

# --- Rate limiting (spec §6.4) ------------------------------------------------
# The Limiter is attached to app.state (slowapi's required lookup point), the
# 429 handler translates RateLimitExceeded into a JSON response, and the
# middleware enforces the app-wide default limit ("60/minute") on every route.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(exercises.router)
app.include_router(workouts.router)
app.include_router(programs.router)
app.include_router(chat.router)


@app.get("/health")
@limiter.exempt
async def health(request: Request) -> JSONResponse:
    """Readiness probe that actually touches the DB.

    The deployed keep-alive ping must reach the database to count as meaningful
    (spec §16), so this runs ``SELECT 1`` through the async engine. Returns 200
    ``{"status": "ok"}`` on success, or 503 ``{"status": "degraded"}`` if the DB
    is unreachable.

    Exempt from rate limiting: keep-alive pings (Render free-tier anti-idle)
    must never be throttled, or the instance could be spun down.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        return JSONResponse(status_code=503, content={"status": "degraded"})
    return JSONResponse(status_code=200, content={"status": "ok"})
