# src/api/main.py

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from api.middleware.logging import log_requests
from api.routers import ingestion, rag, health, status

app = FastAPI(title="IDP Engine API")

# Middleware
app.middleware("http")(log_requests)

# Routers
app.include_router(ingestion.router)
app.include_router(rag.router)
app.include_router(health.router)
app.include_router(status.router)

# Prometheus
Instrumentator().instrument(app).expose(app)
