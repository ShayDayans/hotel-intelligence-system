"""
Direct search for cleanliness reviews using Pinecone API
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# Configuration
PROPERTY_ID = "ABB_40458495"
INDEX_NAME = "airbnb-index"
QUERY = "cleanliness clean spotless tidy"

print("=" * 70)
print(f"SEARCHING REVIEWS FOR {PROPERTY_ID}")
print("=" * 70)
print(f"Query: {QUERY}")
print("=" * 70)

# Load embeddings
print("\nLoading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')

# Generate query embedding
print("Generating query embedding...")
query_embedding = embeddings.embed_query(QUERY)

# Connect to Pinecone
print("\nQuerying Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

# Query with filter
results = index.query(
    vector=query_embedding,
    top_k=10,
    namespace="airbnb_reviews",
    filter={"hotel_id": {"$eq": PROPERTY_ID}},
    include_metadata=True
)

print(f"\n[OK] Found {len(results.matches)} matching reviews")
print("-" * 70)

for i, match in enumerate(results.matches, 1):
    print(f"\n--- Review {i} (Score: {match.score:.3f}) ---")
    print(f"ID: {match.id}")
    print(f"Hotel ID: {match.metadata.get('hotel_id', 'N/A')}")
    print(f"City: {match.metadata.get('city', 'N/A')}")
    
# Also search without filter to see all reviews
print("\n\n" + "=" * 70)
print("ALL REVIEWS IN NAMESPACE (no filter)")
print("=" * 70)

all_results = index.query(
    vector=query_embedding,
    top_k=10,
    namespace="airbnb_reviews",
    include_metadata=True
)

print(f"Found {len(all_results.matches)} total reviews")
for i, match in enumerate(all_results.matches, 1):
    print(f"\n  {i}. {match.id} (score: {match.score:.3f})")
    print(f"     Hotel: {match.metadata.get('hotel_id', 'N/A')}")
