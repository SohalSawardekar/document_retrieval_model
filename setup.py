"""
Setup script for the Hybrid Document Retrieval API.
Run this to download the embedding model and prepare the environment.
"""
from sentence_transformers import SentenceTransformer

def setup():
    print("Downloading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print("\nSetup complete! You can now run the server with:")
    print("  uvicorn main:app --reload")

if __name__ == "__main__":
    setup()
