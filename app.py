from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from graph import app_graph
import os

app = FastAPI(title="Agentic AI RAG API")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: List[str]
    confidence_score: float

@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    if not os.path.exists("./chroma_db"):
        raise HTTPException(status_code=500, detail="Vector DB not found. Please run ingest.py first.")
        
    initial_state = {"query": request.query}
    
    try:
        # Run the graph
        result = app_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return QueryResponse(
        query=result.get("query", request.query),
        final_answer=result.get("final_answer", ""),
        retrieved_context_chunks=result.get("context_chunks", []),
        confidence_score=result.get("confidence_score", 0.0)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
