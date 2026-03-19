"""
Test script for the Legal Judgment Retrieval API.
Can load CSV files and test search functionality.
"""
import requests
import json
import pandas as pd
import ast
import sys

BASE_URL = "http://localhost:8000"


def parse_field(value):
    """Parse a field value - handle list strings or return as string."""
    if pd.isna(value):
        return None

    value_str = str(value).strip()

    # Try to parse as a Python list
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            parsed = ast.literal_eval(value_str)
            if isinstance(parsed, list):
                # Join list items into a single string
                return " ".join(str(item) for item in parsed if item)
        except (ValueError, SyntaxError):
            pass

    return value_str


def load_judgments_from_csv(csv_path: str) -> bool:
    """Load judgments from CSV and index them via API."""
    print(f"\n=== Loading Judgments from CSV: {csv_path} ===")

    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} judgments from CSV")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

    # Prepare judgments
    judgments = []
    for _, row in df.iterrows():
        judgment = {
            "case_no": str(row.get("Case No", "")),
            "title": str(row.get("Title", "")),
            "jurisdiction": str(row.get("Jurisdiction", "")) if pd.notna(row.get("Jurisdiction")) else None,
            "date": str(row.get("Date", "")) if pd.notna(row.get("Date")) else None,
            "issue": parse_field(row.get("Issue")),
            "facts": parse_field(row.get("Facts")),
            "court_reasoning": parse_field(row.get("Court Reasoning")),
            "precedent_analysis": parse_field(row.get("Precedent Analysis")),
            "argument_by_petitioner": parse_field(row.get("Argument by Petitioner")),
            "conclusion": parse_field(row.get("Conclusion")),
            "ipc_sections": str(row.get("ipc_sections", "")) if pd.notna(row.get("ipc_sections")) else None,
            "statute_analysis": parse_field(row.get("Statute Analysis")),
            "argument_by_respondent": parse_field(row.get("Argument by Respondent")),
        }
        judgments.append(judgment)

    # Index via API in batches
    batch_size = 50
    total_indexed = 0

    for i in range(0, len(judgments), batch_size):
        batch = judgments[i:i + batch_size]
        response = requests.post(
            f"{BASE_URL}/judgments/index",
            json={"judgments": batch}
        )

        if response.status_code == 200:
            total_indexed += len(batch)
            print(f"Indexed batch {i//batch_size + 1}: {len(batch)} judgments")
        else:
            print(f"Error indexing batch {i//batch_size + 1}: {response.text}")
            return False

    print(f"Successfully indexed {total_indexed} judgments!")
    return True


def test_index_sample_judgments():
    """Test indexing sample judgments."""
    print("\n=== Testing Sample Judgment Indexing ===")

    judgments = {
        "judgments": [
            {
                "case_no": "2024/001",
                "title": "State vs. John Doe - Murder Trial",
                "jurisdiction": "Supreme Court of India",
                "date": "2024-01-15",
                "issue": "Whether the accused committed murder under Section 302 IPC",
                "facts": "The accused was found with the murder weapon near the crime scene. Witnesses saw him fleeing.",
                "court_reasoning": "The court examined the evidence and found the chain of custody intact.",
                "precedent_analysis": "Following the precedent in Bachan Singh v. State of Punjab",
                "argument_by_petitioner": "The prosecution argued that the circumstantial evidence proves guilt beyond doubt.",
                "conclusion": "The accused is found guilty under Section 302 IPC and sentenced to life imprisonment.",
                "ipc_sections": "302, 201",
                "statute_analysis": "Section 302 IPC defines punishment for murder.",
                "argument_by_respondent": "The defense argued lack of direct evidence and benefit of doubt."
            },
            {
                "case_no": "2024/002",
                "title": "ABC Corp vs. XYZ Ltd - Contract Dispute",
                "jurisdiction": "Delhi High Court",
                "date": "2024-02-20",
                "issue": "Breach of contract and claim for damages",
                "facts": "The defendant failed to deliver goods as per the agreed timeline.",
                "court_reasoning": "The court found the contract terms clear and the breach established.",
                "precedent_analysis": "Referring to Hadley v. Baxendale for damages calculation",
                "argument_by_petitioner": "The plaintiff claimed loss of business opportunity due to delay.",
                "conclusion": "Defendant liable for breach. Damages awarded as claimed.",
                "ipc_sections": "N/A - Civil Case",
                "statute_analysis": "Indian Contract Act, 1872 - Section 73",
                "argument_by_respondent": "The defendant claimed force majeure as defense."
            },
            {
                "case_no": "2024/003",
                "title": "Ravi Kumar vs. Union of India - Constitutional Challenge",
                "jurisdiction": "Supreme Court of India",
                "date": "2024-03-10",
                "issue": "Constitutional validity of the new amendment",
                "facts": "The petitioner challenged the amendment on grounds of violating fundamental rights.",
                "court_reasoning": "The court applied the basic structure doctrine.",
                "precedent_analysis": "Kesavananda Bharati v. State of Kerala was extensively relied upon.",
                "argument_by_petitioner": "The amendment destroys the basic structure of the Constitution.",
                "conclusion": "The amendment upheld as it does not violate the basic structure.",
                "ipc_sections": "N/A - Constitutional Case",
                "statute_analysis": "Constitution of India - Articles 368, 13",
                "argument_by_respondent": "The Parliament has plenary power to amend the Constitution."
            }
        ]
    }

    response = requests.post(f"{BASE_URL}/judgments/index", json=judgments)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200


def test_hybrid_search():
    """Test hybrid search."""
    print("\n=== Testing Hybrid Search ===")

    queries = [
        {
            "query": "murder trial Section 302",
            "top_k": 3,
            "dense_weight": 0.5,
            "sparse_weight": 0.5
        },
        {
            "query": "contract breach damages",
            "top_k": 3,
            "dense_weight": 0.7,
            "sparse_weight": 0.3
        },
        {
            "query": "constitutional amendment fundamental rights",
            "top_k": 3,
            "dense_weight": 0.3,
            "sparse_weight": 0.7
        }
    ]

    for query in queries:
        print(f"\nQuery: '{query['query']}' (dense: {query['dense_weight']}, sparse: {query['sparse_weight']})")
        response = requests.post(f"{BASE_URL}/search", json=query)

        if response.status_code == 200:
            results = response.json()
            print(f"Found {results['total_results']} results:")
            for r in results['results']:
                print(f"  - {r['case_no']}: {r['title']}")
                print(f"    Dense: {r['dense_score']:.4f}, Sparse: {r['sparse_score']:.4f}, Combined: {r['combined_score']:.4f}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


def test_dense_only():
    """Test dense-only search."""
    print("\n=== Testing Dense-Only Search ===")

    query = {
        "query": "criminal conviction life imprisonment",
        "top_k": 3
    }

    response = requests.post(f"{BASE_URL}/search/dense", json=query)
    print(f"Query: '{query['query']}'")

    if response.status_code == 200:
        results = response.json()
        for r in results['results']:
            print(f"  - {r['case_no']}: {r['title']} (score: {r['dense_score']:.4f})")
    else:
        print(f"Error: {response.status_code}")


def test_sparse_only():
    """Test sparse-only search."""
    print("\n=== Testing Sparse-Only Search ===")

    query = {
        "query": "Section 302 IPC",
        "top_k": 3
    }

    response = requests.post(f"{BASE_URL}/search/sparse", json=query)
    print(f"Query: '{query['query']}'")

    if response.status_code == 200:
        results = response.json()
        for r in results['results']:
            print(f"  - {r['case_no']}: {r['title']} (score: {r['sparse_score']:.4f})")
    else:
        print(f"Error: {response.status_code}")


def test_stats():
    """Test stats endpoint."""
    print("\n=== Testing Stats ===")

    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")


def test_get_judgment():
    """Test getting a specific judgment."""
    print("\n=== Testing Get Judgment ===")

    response = requests.get(f"{BASE_URL}/judgments/2024/001")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")


def print_usage():
    """Print usage information."""
    print("""
Usage: python test_api.py [command] [args]

Commands:
  sample          - Index sample judgments and run all tests
  load <csv>      - Load judgments from CSV file
  search <query>  - Search with a custom query
  stats           - Get statistics
  test            - Run tests on existing data

Examples:
  python test_api.py sample
  python test_api.py load data/judgments.csv
  python test_api.py search "murder IPC 302"
  python test_api.py stats
    """)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "sample":
        print("Legal Judgment Retrieval API Test")
        print("=" * 50)

        if test_index_sample_judgments():
            test_hybrid_search()
            test_dense_only()
            test_sparse_only()
            test_stats()
            test_get_judgment()
        else:
            print("Indexing failed. Make sure the server is running!")

        print("\n" + "=" * 50)
        print("Tests completed!")

    elif command == "load":
        if len(sys.argv) < 3:
            print("Error: Please provide CSV file path")
            print("Usage: python test_api.py load <path_to_csv>")
            sys.exit(1)

        csv_path = sys.argv[2]
        if load_judgments_from_csv(csv_path):
            print("\nIndexing complete!")
        else:
            print("\nIndexing failed!")

    elif command == "search":
        if len(sys.argv) < 3:
            print("Error: Please provide search query")
            print("Usage: python test_api.py search \"<your query>\"")
            sys.exit(1)

        query_text = " ".join(sys.argv[2:])
        print(f"\n=== Searching: '{query_text}' ===")

        response = requests.post(f"{BASE_URL}/search", json={
            "query": query_text,
            "top_k": 5,
            "dense_weight": 0.5,
            "sparse_weight": 0.5
        })

        if response.status_code == 200:
            results = response.json()
            print(f"Found {results['total_results']} results:\n")
            for r in results['results']:
                print(f"Case: {r['case_no']}")
                print(f"Title: {r['title']}")
                if r.get('jurisdiction'):
                    print(f"Jurisdiction: {r['jurisdiction']}")
                if r.get('date'):
                    print(f"Date: {r['date']}")
                print(f"Scores: Dense={r['dense_score']:.4f}, Sparse={r['sparse_score']:.4f}, Combined={r['combined_score']:.4f}")
                print("-" * 50)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    elif command == "stats":
        test_stats()

    elif command == "test":
        test_hybrid_search()
        test_dense_only()
        test_sparse_only()
        test_stats()
        test_get_judgment()

    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)
