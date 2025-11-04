# FastAPI app entry point
# - Mount API routes
# - Configure CORS
# - Defines app, middleware, exception handlers
# - Serves endpoints: /benchmark, /results, /health
# - Run with: uvicorn src.api.main:app --reload


# steps to run the project
# 1. Activate your virtual environment
#   On Windows:.venv\Scripts\activate
#   On macOS/Linux: source .venv/bin/activate
# 2. Start the FastAPI server using Uvicorn from project root:
#   uvicorn src.api.main:app --reload

from fastapi import FastAPI, Request
from src.api.routes import dbtest
from src.api.routes import criteria,questions,evidence,chatbot

# Create FastAPI app instance
app = FastAPI(
    title="LLM Benchmarking API",
    description="REST API for benchmarking multiple LLMs models.",
    version="1.0.0"
)


# Include routers for each use-case:
app.include_router(criteria.router, prefix="/benchmark/criteria", tags=["Criteria Generation"])
# app.include_router(questions.router, prefix="/benchmark/questions", tags=["Question Generation"])
# app.include_router(evidence.router, prefix="/benchmark/evidence", tags=["Evidence Analyzer"])
# app.include_router(chatbot.router, prefix="/benchmark/chatbot", tags=["Policy Chatbot"])

# db testing
app.include_router(dbtest.router, prefix="/dbtest", tags=["DatabaseTesting"])

@app.get("/health", tags=["Utility"])
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/", tags=["Utility"])
def root():
    """Root endpoint for API."""
    return {
        "status": "running",
        "message": "LLM Benchmarking API is live. See /docs for OpenAPI documentation."
    }
