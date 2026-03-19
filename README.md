# Legal Judgment Retrieval API

A hybrid document retrieval system for legal judgments combining semantic (dense) and keyword (sparse/BM25) search.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Query    │────▶│  Query Processor │────▶│  Embedding Model│
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                              ┌───────────────────────────┼───────────┐
                              │                           │           │
                              ▼                           ▼           ▼
                    ┌─────────────────┐         ┌─────────────────┐  ┌─────────────────┐
                    │  Sparse Query   │         │  Dense Vector   │  │  Tokenized      │
                    │  (BM25)         │         │  (Embedding)    │  │  Query          │
                    └────────┬────────┘         └────────┬────────┘  └────────┬────────┘
                             │                         │                    │
                             ▼                         ▼                    │
                    ┌─────────────────┐         ┌─────────────────┐         │
                    │  BM25 Index     │         │  ChromaDB       │         │
                    │  (In-Memory)    │         │  (Vector DB)    │         │
                    └────────┬────────┘         └────────┬────────┘         │
                             │                         │                    │
                             │    ┌────────────────────┘                    │
                             │    │                                         │
                             ▼    ▼                                         ▼
                    ┌─────────────────┐                           ┌─────────────────┐
                    │  Sparse Results │                           │  Dense Results  │
                    │  (case_no, score)│                          │  (case_no, score)│
                    └────────┬────────┘                           └────────┬────────┘
                             │                                           │
                             └───────────────────┬───────────────────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────┐
                                    │   Fusion Layer      │
                                    │  - RRF (default)    │
                                    │  - Weighted Sum     │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │   Ranked Results    │
                                    │   (Top-K Merged)    │
                                    └─────────────────────┘
```

## Features

- **Hybrid Search**: Combines dense vector search (ChromaDB + Sentence Transformers) with sparse keyword search (BM25)
- **Legal Document Fields**: Supports all judgment fields:
  - Case No, Title, Jurisdiction, Date
  - Issue, Facts, Court Reasoning
  - Precedent Analysis, Arguments (Petitioner/Respondent)
  - Conclusion, IPC Sections, Statute Analysis
- **List Field Support**: Automatically handles CSV columns containing lists (e.g., `['item1', 'item2']`)
- **FastAPI Backend**: RESTful API with automatic documentation at `/docs`

## Setup

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Embedding Model

```bash
python setup.py
```

### 4. Start the Server

```bash
# Development
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Usage Commands

### Test with Sample Data

```bash
python test_api.py sample
```

Indexes sample judgments and runs all tests.

### Load Judgments from CSV

```bash
python test_api.py load <path_to_csv>
```

Example:
```bash
python test_api.py load data/judgments.csv
```

**CSV Format**: Your CSV should have these columns:
- `Case No` (required) - Unique identifier
- `Title` (required) - Case title
- `Jurisdiction` - Court jurisdiction
- `Date` - Judgment date
- `Issue` - Legal issue(s)
- `Facts` - Case facts
- `Court Reasoning` - Court's reasoning
- `Precedent Analysis` - Analysis of precedents
- `Argument by Petitioner` - Petitioner's arguments
- `Conclusion` - Final conclusion
- `ipc_sections` - Applicable IPC sections
- `Statute Analysis` - Analysis of statutes
- `Argument by Respondent` - Respondent's arguments

**Note**: Fields can be strings or lists of strings (e.g., `"['item1', 'item2']"`).

### Search Judgments

```bash
python test_api.py search "your query here"
```

Example:
```bash
python test_api.py search "murder IPC 302"
```

### Get Statistics

```bash
python test_api.py stats
```

### Run Tests on Existing Data

```bash
python test_api.py test
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/judgments/index` | Index a batch of judgments |
| POST | `/search` | Hybrid search (dense + sparse) |
| POST | `/search/dense` | Dense-only search (semantic) |
| POST | `/search/sparse` | Sparse-only search (BM25) |
| GET | `/judgments/{case_no}` | Get a specific judgment |
| DELETE | `/judgments/{case_no}` | Delete a judgment |
| GET | `/stats` | Get index statistics |

## Search Examples

### Hybrid Search (curl)

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "murder IPC 302",
    "top_k": 5,
    "dense_weight": 0.5,
    "sparse_weight": 0.5
  }'
```

### Dense-Only Search

```bash
curl -X POST http://localhost:8000/search/dense \
  -H "Content-Type: application/json" \
  -d '{
    "query": "constitutional fundamental rights",
    "top_k": 10
  }'
```

### Sparse-Only Search

```bash
curl -X POST http://localhost:8000/search/sparse \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Section 420 IPC",
    "top_k": 10
  }'
```

### Index Judgments (curl)

```bash
curl -X POST http://localhost:8000/judgments/index \
  -H "Content-Type: application/json" \
  -d '{
    "judgments": [
      {
        "case_no": "2024/001",
        "title": "State vs. John Doe",
        "jurisdiction": "Supreme Court",
        "issue": "Murder under Section 302 IPC",
        "facts": "Accused found with weapon...",
        "conclusion": "Guilty, sentenced to life imprisonment",
        "ipc_sections": "302, 201"
      }
    ]
  }'
```

### Get a Judgment

```bash
curl http://localhost:8000/judgments/2024/001
```

### Delete a Judgment

```bash
curl -X DELETE http://localhost:8000/judgments/2024/001
```

## Search Parameters

The `/search` endpoint accepts:

```json
{
  "query": "your search query",
  "top_k": 10,           // Number of results (default: 10)
  "dense_weight": 0.5,   // Weight for semantic search (0-1)
  "sparse_weight": 0.5   // Weight for keyword search (0-1)
}
```

- When `dense_weight == sparse_weight == 0.5`, uses Reciprocal Rank Fusion (RRF)
- Otherwise, uses weighted sum of normalized scores

## Response Format

```json
{
  "results": [
    {
      "case_no": "2024/001",
      "title": "State vs. John Doe",
      "jurisdiction": "Supreme Court",
      "date": "2024-01-15",
      "issue": "Murder under Section 302 IPC",
      "facts": "Case facts here...",
      "court_reasoning": "Court reasoning here...",
      "conclusion": "Guilty",
      "ipc_sections": "302",
      "dense_score": 0.8234,
      "sparse_score": 1.2345,
      "combined_score": 0.0456
    }
  ],
  "query": "murder IPC 302",
  "total_results": 5
}
```

## Project Structure

```
.
├── main.py              # FastAPI application
├── test_api.py          # CLI tool for testing and CSV loading
├── setup.py             # Model download script
├── requirements.txt     # Python dependencies
├── chroma_data/         # ChromaDB persistent storage
├── .gitignore          # Git ignore file
└── README.md           # This file
```

## How It Works

1. **Indexing**: Text fields are combined and embedded using `all-MiniLM-L6-v2`. Embeddings are stored in ChromaDB, and tokenized text is indexed in BM25.

2. **Dense Search**: Query is embedded and similarity search is performed in ChromaDB using cosine distance.

3. **Sparse Search**: Query is tokenized and scored using BM25 against the indexed documents.

4. **Fusion**: Results from both searches are combined using Reciprocal Rank Fusion (RRF) or weighted sum.

## Troubleshooting

### "Indexing failed" error
- Make sure the server is running: `uvicorn main:app --reload`
- Check that port 8000 is not in use

### CSV parsing errors
- Ensure CSV encoding is UTF-8
- List fields should be formatted as: `['item1', 'item2']` or `"['item1', 'item2']"`

### ChromaDB errors
- Delete the `chroma_data/` folder and restart to reset the database

### Import errors
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again

## Dependencies

- **FastAPI**: Web framework
- **ChromaDB**: Vector database (persistent local storage)
- **sentence-transformers**: Embedding generation (all-MiniLM-L6-v2)
- **rank-bm25**: BM25 implementation for keyword search
- **pandas**: CSV loading and processing
- **pydantic**: Data validation
- **uvicorn**: ASGI server

