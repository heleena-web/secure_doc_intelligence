from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.errors import AppException
from app.storage.database import init_db
from app.api.v1.documents import router as documents_router
from app.api.v1.search import router as search_router
from app.api.v1.qa import router as qa_router
from app.api.v1.auth import router as auth_router
from app.api.v1.notion import router as notion_router


async def sync_search_indices() -> None:
    """Restores in-memory BM25 index and synchronizes ChromaDB vectors for all indexed documents on startup."""
    try:
        from app.storage.database import get_db_connection
        from app.storage.repository import DocumentRepository
        from app.services.retrieval.search_service import search_service
        from app.models.document import IngestionStatus

        async with get_db_connection() as db:
            repo = DocumentRepository(db)
            cursor = await db.execute("SELECT id, filename, status FROM documents")
            docs = await cursor.fetchall()
            for d in docs:
                chunks = await repo.get_chunks_by_document(d["id"])
                if chunks:
                    if d["status"] != IngestionStatus.INDEXED.value:
                        await repo.update_document_status(
                            document_id=d["id"],
                            status=IngestionStatus.INDEXED,
                            chunk_count=len(chunks),
                        )
                    search_service.index_document_chunks(chunks, d["filename"])
    except Exception as e:
        print(f"Warning: Failed to sync search indices on startup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes authoritative SQLite database schema and restores search indices on startup."""
    await init_db()
    await sync_search_indices()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Privacy-focused, permission-aware document intelligence platform.",
    lifespan=lifespan,
)

# CORS middleware for clean frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi.exceptions import RequestValidationError


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Translates custom application exceptions into clean JSON error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "detail": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Translates request validation errors into clean human-readable detail strings."""
    messages = []
    for err in exc.errors():
        loc_parts = [str(x) for x in err.get("loc", []) if str(x) != "body"]
        field = " -> ".join(loc_parts)
        msg = err.get("msg", "Invalid value")
        messages.append(f"{field}: {msg}" if field else msg)
    detail_str = "; ".join(messages) if messages else "Invalid request data"
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "detail": detail_str,
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catches unexpected server errors and returns informative JSON."""
    import traceback
    tb = traceback.format_exc()
    print("UNHANDLED EXCEPTION:\n", tb)
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.__class__.__name__,
            "detail": str(exc),
            "traceback": tb,
        },
    )



# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }


# Include API Routers
app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(documents_router, prefix=settings.API_V1_PREFIX)
app.include_router(search_router, prefix=settings.API_V1_PREFIX)
app.include_router(qa_router, prefix=settings.API_V1_PREFIX)
app.include_router(notion_router, prefix=settings.API_V1_PREFIX)
