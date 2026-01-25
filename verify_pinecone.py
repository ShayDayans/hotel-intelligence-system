"""Verify what's in Pinecone for ABB_40458495"""
import os
from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("airbnb-index")

# Get index stats
stats = index.describe_index_stats()
print("Index stats:")
print(f"  Total vectors: {stats.total_vector_count}")
print(f"  Namespaces: {stats.namespaces}")

# Try to fetch the specific reviews we uploaded
print("\nFetching uploaded reviews...")
result = index.fetch(
    ids=["ABB_40458495_R0", "ABB_40458495_R1", "ABB_40458495_R2"],
    namespace="airbnb_reviews"
)
print(f"Found {len(result.vectors)} vectors")
for vid, vec in result.vectors.items():
    print(f"\n  ID: {vid}")
    print(f"  Metadata: {vec.metadata}")

# Also check the property in airbnb_hotels
print("\n\nFetching property from airbnb_hotels...")
result2 = index.fetch(
    ids=["ABB_40458495"],
    namespace="airbnb_hotels"
)
print(f"Found {len(result2.vectors)} vectors")
for vid, vec in result2.vectors.items():
    print(f"  ID: {vid}")
    print(f"  Metadata: {vec.metadata}")
