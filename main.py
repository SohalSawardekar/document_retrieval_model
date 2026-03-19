from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import asyncio
from concurrent.futures import ThreadPoolExecutor
from chromadb.config import Settings

app = FastAPI(title="Legal Judgment Retrieval API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Initialize ChromaDB PersistentClient for local storage
client = chromadb.PersistentClient(
    path="./chroma_data", settings=Settings(anonymized_telemetry=False)
)

# Collection for judgments
collection = client.get_or_create_collection(
    name="judgements", metadata={"hnsw:space": "cosine"}
)

# In-memory storage for sparse retrieval
# Structure: {case_no: {"text": str, "tokens": List[str], "judgment_data": dict}}
document_store: Dict[str, Dict[str, Any]] = {}
bm25_index: Optional[BM25Okapi] = None
doc_ids_list: List[str] = []

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)


# Pydantic models for Legal Judgments
class Judgment(BaseModel):
    case_no: str
    title: str
    jurisdiction: Optional[str] = None
    date: Optional[str] = None
    issue: Optional[str] = None
    facts: Optional[str] = None
    court_reasoning: Optional[str] = None
    precedent_analysis: Optional[str] = None
    argument_by_petitioner: Optional[str] = None
    conclusion: Optional[str] = None
    ipc_sections: Optional[str] = None
    statute_analysis: Optional[str] = None
    argument_by_respondent: Optional[str] = None

    def get_combined_text(self) -> str:
        """Combine all text fields for embedding."""
        fields = [
            self.title,
            self.issue,
            self.facts,
            self.court_reasoning,
            self.precedent_analysis,
            self.argument_by_petitioner,
            self.conclusion,
            self.statute_analysis,
            self.argument_by_respondent,
        ]
        # Handle both str and list[str]
        text_parts = []
        for field in fields:
            if field is None:
                continue
            if isinstance(field, list):
                text_parts.extend(str(item) for item in field if item)
            else:
                text_parts.append(str(field))
        return " ".join(text_parts)


class JudgmentBatch(BaseModel):
    judgments: List[Judgment]


class SearchQuery(BaseModel):
    query: str
    top_k: int = 10
    dense_weight: float = 0.5
    sparse_weight: float = 0.5


class SearchResult(BaseModel):
    case_no: str
    title: str
    jurisdiction: Optional[str] = None
    date: Optional[str] = None
    issue: Optional[str] = None
    facts: Optional[str] = None
    court_reasoning: Optional[str] = None
    precedent_analysis: Optional[str] = None
    argument_by_petitioner: Optional[str] = None
    conclusion: Optional[str] = None
    ipc_sections: Optional[str] = None
    statute_analysis: Optional[str] = None
    argument_by_respondent: Optional[str] = None
    dense_score: float
    sparse_score: float
    combined_score: float

    class Config:
        # Allow list fields to be accepted
        arbitrary_types_allowed = True


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int


# Helper functions
def combine_text_fields(judgment_data: dict) -> str:
    """Combine all text fields for embedding, handling both str and list[str]."""
    fields = [
        judgment_data.get("title", ""),
        judgment_data.get("issue"),
        judgment_data.get("facts"),
        judgment_data.get("court_reasoning"),
        judgment_data.get("precedent_analysis"),
        judgment_data.get("argument_by_petitioner"),
        judgment_data.get("conclusion"),
        judgment_data.get("statute_analysis"),
        judgment_data.get("argument_by_respondent"),
    ]
    # Handle both str and list[str]
    text_parts = []
    for field in fields:
        if field is None:
            continue
        if isinstance(field, list):
            text_parts.extend(str(item) for item in field if item)
        else:
            text_parts.append(str(field))
    return " ".join(text_parts)


def tokenize_text(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    return text.lower().split()


def compute_embeddings(texts: List[str]) -> np.ndarray:
    """Compute embeddings for a list of texts."""
    return embedding_model.encode(texts, show_progress_bar=False)


def rebuild_bm25_index():
    """Rebuild BM25 index from document store."""
    global bm25_index, doc_ids_list

    if not document_store:
        bm25_index = None
        doc_ids_list = []
        return

    doc_ids_list = list(document_store.keys())
    tokenized_docs = [document_store[doc_id]["tokens"] for doc_id in doc_ids_list]
    bm25_index = BM25Okapi(tokenized_docs)


def reciprocal_rank_fusion(
    dense_results: Dict[str, float], sparse_results: Dict[str, float], k: int = 60
) -> Dict[str, float]:
    """Combine dense and sparse results using Reciprocal Rank Fusion."""
    fused_scores = {}

    sorted_dense = sorted(dense_results.items(), key=lambda x: x[1], reverse=True)
    for rank, (doc_id, score) in enumerate(sorted_dense, start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (k + rank)

    sorted_sparse = sorted(sparse_results.items(), key=lambda x: x[1], reverse=True)
    for rank, (doc_id, score) in enumerate(sorted_sparse, start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (k + rank)

    return fused_scores


def weighted_fusion(
    dense_results: Dict[str, float],
    sparse_results: Dict[str, float],
    dense_weight: float,
    sparse_weight: float,
) -> Dict[str, float]:
    """Combine dense and sparse results using weighted sum."""
    all_doc_ids = set(dense_results.keys()) | set(sparse_results.keys())

    dense_max = max(dense_results.values()) if dense_results else 1.0
    sparse_max = max(sparse_results.values()) if sparse_results else 1.0

    fused_scores = {}
    for doc_id in all_doc_ids:
        dense_score = dense_results.get(doc_id, 0) / dense_max
        sparse_score = sparse_results.get(doc_id, 0) / sparse_max
        fused_scores[doc_id] = (dense_weight * dense_score) + (
            sparse_weight * sparse_score
        )

    return fused_scores


# API Endpoints
@app.post("/judgments/index", response_model=Dict[str, Any])
async def index_judgments(batch: JudgmentBatch):
    """Index a batch of legal judgments for hybrid search."""
    try:
        texts = []
        ids = []
        metadatas = []

        for judgment in batch.judgments:
            # Combine text fields for embedding
            combined_text = combine_text_fields(judgment.model_dump())
            tokens = tokenize_text(combined_text)

            # Store in document store for BM25
            document_store[judgment.case_no] = {
                "text": combined_text,
                "tokens": tokens,
                "judgment": judgment.model_dump(),
            }

            texts.append(combined_text)
            ids.append(judgment.case_no)
            # ChromaDB metadata only supports str, int, float, bool - filter out None
            metadata = {
                "title": judgment.title,
            }
            if judgment.jurisdiction:
                metadata["jurisdiction"] = judgment.jurisdiction
            if judgment.date:
                metadata["date"] = judgment.date
            if judgment.ipc_sections:
                metadata["ipc_sections"] = judgment.ipc_sections
            metadatas.append(metadata)

        # Compute embeddings in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(executor, compute_embeddings, texts)

        # Add to ChromaDB
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        # Rebuild BM25 index
        rebuild_bm25_index()

        return {
            "status": "success",
            "message": f"Indexed {len(batch.judgments)} judgments",
            "total_judgments": len(document_store),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def hybrid_search(search_query: SearchQuery):
    """Perform hybrid search combining dense (semantic) and sparse (keyword) retrieval."""
    try:
        query = search_query.query
        top_k = search_query.top_k

        loop = asyncio.get_event_loop()

        dense_task = loop.run_in_executor(
            executor, lambda: dense_search(query, top_k * 2)
        )
        sparse_task = loop.run_in_executor(
            executor, lambda: sparse_search(query, top_k * 2)
        )

        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        if search_query.dense_weight == 0.5 and search_query.sparse_weight == 0.5:
            fused_scores = reciprocal_rank_fusion(dense_results, sparse_results)
        else:
            fused_scores = weighted_fusion(
                dense_results,
                sparse_results,
                search_query.dense_weight,
                search_query.sparse_weight,
            )

        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        results = []
        for case_no, combined_score in sorted_results:
            doc_data = document_store.get(case_no, {})
            judgment_data = doc_data.get("judgment", {})
            results.append(
                SearchResult(
                    case_no=case_no,
                    title=judgment_data.get("title", ""),
                    jurisdiction=judgment_data.get("jurisdiction"),
                    date=judgment_data.get("date"),
                    issue=judgment_data.get("issue"),
                    facts=judgment_data.get("facts"),
                    court_reasoning=judgment_data.get("court_reasoning"),
                    precedent_analysis=judgment_data.get("precedent_analysis"),
                    argument_by_petitioner=judgment_data.get("argument_by_petitioner"),
                    conclusion=judgment_data.get("conclusion"),
                    ipc_sections=judgment_data.get("ipc_sections"),
                    statute_analysis=judgment_data.get("statute_analysis"),
                    argument_by_respondent=judgment_data.get("argument_by_respondent"),
                    dense_score=dense_results.get(case_no, 0.0),
                    sparse_score=sparse_results.get(case_no, 0.0),
                    combined_score=combined_score,
                )
            )

        return SearchResponse(results=results, query=query, total_results=len(results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def dense_search(query: str, top_k: int) -> Dict[str, float]:
    """Perform dense vector search using ChromaDB."""
    query_embedding = embedding_model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, len(document_store) or 1),
        include=["distances"],
    )

    scores = {}
    if results["ids"] and results["ids"][0]:
        for case_no, distance in zip(results["ids"][0], results["distances"][0]):
            scores[case_no] = 1.0 - float(distance)

    return scores


def sparse_search(query: str, top_k: int) -> Dict[str, float]:
    """Perform sparse keyword search using BM25."""
    if bm25_index is None or not document_store:
        return {}

    query_tokens = tokenize_text(query)
    scores = bm25_index.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = {}
    for idx in top_indices:
        if scores[idx] > 0:
            case_no = doc_ids_list[idx]
            results[case_no] = float(scores[idx])

    return results


@app.get("/judgments/{case_no}", response_model=Judgment)
async def get_judgment(case_no: str):
    """Retrieve a specific judgment by case number."""
    if case_no not in document_store:
        raise HTTPException(status_code=404, detail="Judgment not found")

    doc_data = document_store[case_no]
    return Judgment(**doc_data.get("judgment", {}))


@app.delete("/judgments/{case_no}")
async def delete_judgment(case_no: str):
    """Delete a judgment by case number."""
    try:
        if case_no not in document_store:
            raise HTTPException(status_code=404, detail="Judgment not found")

        del document_store[case_no]
        collection.delete(ids=[case_no])
        rebuild_bm25_index()

        return {"status": "success", "message": f"Judgment {case_no} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get statistics about the indexed judgments."""
    return {
        "total_judgments": len(document_store),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": embedding_model.get_sentence_embedding_dimension(),
        "has_bm25_index": bm25_index is not None,
    }


@app.post("/search/dense", response_model=SearchResponse)
async def dense_search_only(search_query: SearchQuery):
    """Search using only dense (semantic) retrieval."""
    try:
        loop = asyncio.get_event_loop()
        dense_results = await loop.run_in_executor(
            executor, lambda: dense_search(search_query.query, search_query.top_k)
        )

        results = []
        for case_no, score in sorted(
            dense_results.items(), key=lambda x: x[1], reverse=True
        ):
            doc_data = document_store.get(case_no, {})
            judgment_data = doc_data.get("judgment", {})
            results.append(
                SearchResult(
                    case_no=case_no,
                    title=judgment_data.get("title", ""),
                    jurisdiction=judgment_data.get("jurisdiction"),
                    date=judgment_data.get("date"),
                    issue=judgment_data.get("issue"),
                    facts=judgment_data.get("facts"),
                    court_reasoning=judgment_data.get("court_reasoning"),
                    precedent_analysis=judgment_data.get("precedent_analysis"),
                    argument_by_petitioner=judgment_data.get("argument_by_petitioner"),
                    conclusion=judgment_data.get("conclusion"),
                    ipc_sections=judgment_data.get("ipc_sections"),
                    statute_analysis=judgment_data.get("statute_analysis"),
                    argument_by_respondent=judgment_data.get("argument_by_respondent"),
                    dense_score=score,
                    sparse_score=0.0,
                    combined_score=score,
                )
            )

        return SearchResponse(
            results=results, query=search_query.query, total_results=len(results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/sparse", response_model=SearchResponse)
async def sparse_search_only(search_query: SearchQuery):
    """Search using only sparse (keyword/BM25) retrieval."""
    try:
        loop = asyncio.get_event_loop()
        sparse_results = await loop.run_in_executor(
            executor, lambda: sparse_search(search_query.query, search_query.top_k)
        )

        results = []
        for case_no, score in sorted(
            sparse_results.items(), key=lambda x: x[1], reverse=True
        ):
            doc_data = document_store.get(case_no, {})
            judgment_data = doc_data.get("judgment", {})
            results.append(
                SearchResult(
                    case_no=case_no,
                    title=judgment_data.get("title", ""),
                    jurisdiction=judgment_data.get("jurisdiction"),
                    date=judgment_data.get("date"),
                    issue=judgment_data.get("issue"),
                    facts=judgment_data.get("facts"),
                    court_reasoning=judgment_data.get("court_reasoning"),
                    precedent_analysis=judgment_data.get("precedent_analysis"),
                    argument_by_petitioner=judgment_data.get("argument_by_petitioner"),
                    conclusion=judgment_data.get("conclusion"),
                    ipc_sections=judgment_data.get("ipc_sections"),
                    statute_analysis=judgment_data.get("statute_analysis"),
                    argument_by_respondent=judgment_data.get("argument_by_respondent"),
                    dense_score=0.0,
                    sparse_score=score,
                    combined_score=score,
                )
            )

        return SearchResponse(
            results=results, query=search_query.query, total_results=len(results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
