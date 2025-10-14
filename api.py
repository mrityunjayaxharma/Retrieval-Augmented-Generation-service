"""FastAPI application for RAG service"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime
from loguru import logger

from config import settings
from crawler import WebCrawler
from indexer import DocumentIndexer
from qa_system import QuestionAnsweringSystem
from utils import setup_logging

setup_logging("INFO", f"{settings.logs_dir}/rag_service.log")

class CrawlRequest(BaseModel):
    start_url: HttpUrl
    max_pages: Optional[int] = Field(default=None, ge=1, le=100)
    max_depth: Optional[int] = Field(default=None, ge=1, le=5)
    crawl_delay_ms: Optional[int] = Field(default=None, ge=100, le=10000)

class IndexRequest(BaseModel):
    chunk_size: Optional[int] = Field(default=None, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=500)
    embedding_model: Optional[str] = Field(default=None)

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)

class CrawlResponse(BaseModel):
    page_count: int
    skipped_count: int
    failed_count: int
    urls: List[str]
    stats: Dict[str, Any]
    domain: str

class IndexResponse(BaseModel):
    total_documents: int
    total_chunks: int
    index_info: Dict[str, Any]
    total_time_ms: float
    chunks_per_doc: float

class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    timings: Dict[str, float]
    is_answerable: bool
    refusal_reason: Optional[str]

class StatusResponse(BaseModel):
    status: str
    message: str
    system_info: Dict[str, Any]
    timestamp: str

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Retrieval-Augmented Generation (RAG) service for website content",
    docs_url="/docs",
    redoc_url="/redoc"
)

crawler_state = {"is_running": False, "last_crawl": None}
indexer_state = {"is_running": False, "last_index": None}
qa_system = QuestionAnsweringSystem()

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")

@app.get("/", response_model=StatusResponse)
async def root():
    system_status = qa_system.get_system_status()

    return StatusResponse(
        status="running",
        message=f"{settings.app_name} is running",
        system_info=system_status,
        timestamp=datetime.now().isoformat()
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/crawl", response_model=CrawlResponse)
async def crawl_website(request: CrawlRequest, background_tasks: BackgroundTasks):
    if crawler_state["is_running"]:
        raise HTTPException(status_code=409, detail="Crawl already in progress")

    try:
        logger.info(f"Starting crawl for: {request.start_url}")
        crawler_state["is_running"] = True

        crawler = WebCrawler(
            max_pages=request.max_pages or settings.max_pages,
            max_depth=request.max_depth or settings.max_depth,
            crawl_delay_ms=request.crawl_delay_ms or settings.crawl_delay_ms
        )

        crawl_results = crawler.crawl(str(request.start_url))

        crawler_state["last_crawl"] = crawl_results

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crawl_file = f"{settings.data_dir}/crawl_{timestamp}.json"
        crawler.save_crawl_results(crawl_file)

        response = CrawlResponse(
            page_count=crawl_results["stats"]["pages_crawled"],
            skipped_count=crawl_results["stats"]["pages_skipped"],
            failed_count=crawl_results["stats"]["pages_failed"],
            urls=[page.url for page in crawl_results["pages"]],
            stats=crawl_results["stats"],
            domain=crawl_results["domain"]
        )

        logger.info(f"Crawl completed: {response.page_count} pages")
        return response

    except Exception as e:
        logger.error(f"Crawl failed: {e}")
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")

    finally:
        crawler_state["is_running"] = False

@app.post("/index", response_model=IndexResponse)
async def index_content(request: IndexRequest):
    if indexer_state["is_running"]:
        raise HTTPException(status_code=409, detail="Indexing already in progress")

    if not crawler_state["last_crawl"]:
        raise HTTPException(status_code=400, detail="No crawled content available. Run /crawl first.")

    try:
        logger.info("Starting content indexing")
        indexer_state["is_running"] = True

        indexer = DocumentIndexer(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            embedding_model=request.embedding_model
        )

        crawl_results = crawler_state["last_crawl"]
        url_to_content = {page.url: page.content for page in crawl_results["pages"]}

        index_results = indexer.index_documents(url_to_content)

        global qa_system
        qa_system = QuestionAnsweringSystem(indexer)

        indexer_state["last_index"] = index_results

        response = IndexResponse(
            total_documents=index_results["total_documents"],
            total_chunks=index_results["total_chunks"],
            index_info=index_results["index_info"],
            total_time_ms=index_results["total_time_ms"],
            chunks_per_doc=index_results["chunks_per_doc"]
        )

        logger.info(f"Indexing completed: {response.total_chunks} chunks from {response.total_documents} documents")
        return response

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    finally:
        indexer_state["is_running"] = False

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        logger.info(f"Processing question: {request.question[:50]}...")

        system_status = qa_system.get_system_status()
        if not system_status.get("index_info", {}).get("has_index", False):
            raise HTTPException(
                status_code=400,
                detail="No indexed content available. Run /crawl and /index first."
            )

        result = qa_system.answer_question(
            question=request.question,
            top_k=request.top_k
        )

        response = QuestionResponse(
            question=result.question,
            answer=result.answer,
            confidence=result.confidence,
            sources=result.sources,
            timings={
                "retrieval_ms": result.retrieval_time_ms,
                "generation_ms": result.generation_time_ms,
                "total_ms": result.total_time_ms
            },
            is_answerable=result.is_answerable,
            refusal_reason=result.refusal_reason
        )

        logger.info(f"Question answered: confidence={response.confidence:.2f}, answerable={response.is_answerable}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_detailed_status():
    system_status = qa_system.get_system_status()

    system_status.update({
        "crawler": {
            "is_running": crawler_state["is_running"],
            "last_crawl_available": crawler_state["last_crawl"] is not None,
            "last_crawl_pages": len(crawler_state["last_crawl"]["pages"]) if crawler_state["last_crawl"] else 0
        },
        "indexer": {
            "is_running": indexer_state["is_running"],
            "last_index_available": indexer_state["last_index"] is not None
        }
    })

    return StatusResponse(
        status=system_status["status"],
        message=f"System is {system_status['status']}",
        system_info=system_status,
        timestamp=datetime.now().isoformat()
    )

@app.get("/metrics")
async def get_metrics():
    return {
        "crawler_stats": crawler_state["last_crawl"]["stats"] if crawler_state["last_crawl"] else None,
        "index_stats": indexer_state["last_index"] if indexer_state["last_index"] else None,
        "system_status": qa_system.get_system_status()
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )
