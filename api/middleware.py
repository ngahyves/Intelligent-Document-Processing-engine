import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from src.config.logging_config import get_logger

logger=get_logger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request, its duration, and captures any unhandled exceptions.
    Essential for Prometheus monitoring later.
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            process_time = (time.perf_counter() - start_time) * 1000
            
            # Log the request details
            logger.info(
                f"Method: {request.method} | Path: {request.url.path} | "
                f"Status: {response.status_code} | Latency: {process_time:.2f}ms"
            )
            
            # Add custom header with latency
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"CRITICAL ERROR: {str(e)} | Latency: {process_time:.2f}ms")
            
            # Return a structured error instead of crashing
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={"detail": "An internal server error occurred. Please check logs."}
            )