"""
Debug: Why isn't RAG finding location reviews for ABB_40458495?
"""
import os
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

PROPERTY_ID = "ABB_40458495"
INDEX_NAME = "airbnb-index"

print("=" * 70)
print("DEBUG: RAG SEARCH FOR LOCATION REVIEWS")
print("=" * 70)

# 1. Check what's stored in Pinecone
print("\n[1] Checking Pinecone vectors directly...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

# Fetch our reviews
result = index.fetch(
    ids=["ABB_40458495_R0", "ABB_40458495_R1", "ABB_40458495_R2"],
    namespace="airbnb_reviews"
)

print(f"Found {len(result.vectors)} vectors in airbnb_reviews")
for vid, vec in result.vectors.items():
    print(f"\n  {vid}:")
    print(f"    Metadata keys: {list(vec.metadata.keys())}")
    # Check if 'text' key exists (needed for LangChain retrieval)
    if 'text' in vec.metadata:
        print(f"    text: {vec.metadata['text'][:100]}...")
    else:
        print(f"    [WARNING] No 'text' key in metadata! LangChain can't retrieve content.")
    print(f"    hotel_id: {vec.metadata.get('hotel_id')}")

# 2. Try LangChain search
print("\n" + "=" * 70)
print("[2] Testing LangChain PineconeVectorStore search...")
print("=" * 70)

embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace="airbnb_reviews"
)

# Search for location
print("\nSearching for 'location' with hotel_id filter...")
results = vectorstore.similarity_search(
    "location",
    k=5,
    filter={"hotel_id": PROPERTY_ID}
)

print(f"Found {len(results)} results")
for i, doc in enumerate(results):
    print(f"\n  Result {i+1}:")
    print(f"    page_content: {doc.page_content[:100] if doc.page_content else '[EMPTY]'}...")
    print(f"    metadata: {doc.metadata}")

# 3. Try search WITHOUT filter
print("\n" + "=" * 70)
print("[3] Testing search WITHOUT hotel_id filter...")
print("=" * 70)

results_no_filter = vectorstore.similarity_search("location", k=5)
print(f"Found {len(results_no_filter)} results (no filter)")
for i, doc in enumerate(results_no_filter):
    print(f"\n  Result {i+1}:")
    print(f"    page_content: {doc.page_content[:100] if doc.page_content else '[EMPTY]'}...")
    print(f"    hotel_id: {doc.metadata.get('hotel_id', 'N/A')}")

# 4. Direct Pinecone query with embedding
print("\n" + "=" * 70)
print("[4] Direct Pinecone query (bypassing LangChain)...")
print("=" * 70)

query_embedding = embeddings.embed_query("location perfect amazing")
query_result = index.query(
    vector=query_embedding,
    top_k=5,
    namespace="airbnb_reviews",
    filter={"hotel_id": {"$eq": PROPERTY_ID}},
    include_metadata=True
)

print(f"Found {len(query_result.matches)} matches")
for match in query_result.matches:
    print(f"\n  {match.id} (score: {match.score:.3f})")
    print(f"    metadata: {match.metadata}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)
