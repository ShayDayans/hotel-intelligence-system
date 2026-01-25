"""
Re-ingest ALL reviews for ABB_40458495 with 'text' field in metadata
"""
import os
import sys
import json
import ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# Configuration
PROPERTY_ID = "ABB_40458495"
INDEX_NAME = "airbnb-index"
DATA_FILE = "data/airbnb_sampled_three_cities.parquet"

print("=" * 70)
print(f"RE-INGESTING ALL REVIEWS FOR {PROPERTY_ID}")
print("=" * 70)

# Load parquet
print("\n[1/4] Loading data...")
df = pd.read_parquet(DATA_FILE)
row = df[df['property_id'].astype(str).str.contains('40458495')].iloc[0]
reviews_str = row['reviews']
prop_name = str(row.get('name', 'Unknown')).encode('ascii', 'replace').decode('ascii')
location = str(row.get('location', 'Broadbeach')).encode('ascii', 'replace').decode('ascii')

# Parse ALL reviews properly
print("\n[2/4] Parsing reviews...")
reviews = []
try:
    parsed = json.loads(reviews_str)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                text = item.get('review') or item.get('text') or item.get('comment') or item.get('comments') or str(item)
                reviews.append(str(text))
            else:
                reviews.append(str(item))
    elif isinstance(parsed, str):
        reviews = [parsed]
except Exception:
    try:
        parsed = ast.literal_eval(reviews_str)
        if isinstance(parsed, list):
            reviews = [str(x) for x in parsed]
    except Exception:
        reviews = [reviews_str]

print(f"[OK] Parsed {len(reviews)} reviews")

# Preview first 3
for i, r in enumerate(reviews[:3], 1):
    r_clean = r[:80].encode('ascii', 'replace').decode('ascii')
    print(f"  {i}. {r_clean}...")

# Load embeddings
print("\n[3/4] Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')

# Connect to Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

# Delete old reviews first
print("\n[4/4] Uploading reviews to Pinecone...")
print("  Deleting old reviews...")
old_ids = [f"{PROPERTY_ID}_R{i}" for i in range(100)]  # Delete up to 100 old ones
try:
    index.delete(ids=old_ids, namespace="airbnb_reviews")
except:
    pass

# Upload ALL reviews with TEXT in metadata
uploaded = 0
for idx, review_text in enumerate(reviews):
    if not review_text or len(str(review_text).strip()) < 10:
        continue
    
    review_id = f"{PROPERTY_ID}_R{idx}"
    
    # Clean review text
    review_clean = str(review_text).encode('ascii', 'replace').decode('ascii')
    
    # Build content for embedding (same format as ingestion.py)
    content = f"Review for {prop_name} in {location}: {review_clean}"
    
    # CRITICAL: Include 'text' field in metadata for LangChain!
    metadata = {
        "source": "airbnb",
        "review_id": review_id,
        "hotel_id": PROPERTY_ID,
        "hotel_name": prop_name,
        "city": location,
        "country": str(row.get('country', '')).encode('ascii', 'replace').decode('ascii'),
        "text": content,  # <-- THIS IS THE FIX!
    }
    
    # Generate embedding
    embedding = embeddings.embed_documents([content])[0]
    
    # Upload
    vector = {
        "id": review_id,
        "values": embedding,
        "metadata": metadata
    }
    
    index.upsert(vectors=[vector], namespace="airbnb_reviews")
    uploaded += 1
    
    if uploaded <= 5 or uploaded % 5 == 0:
        print(f"  [OK] Uploaded {uploaded}: {review_id}")

print(f"\n{'=' * 70}")
print(f"[SUCCESS] Ingested {uploaded} reviews with 'text' field!")
print(f"{'=' * 70}")
print(f"LangChain RAG should now work properly.")
print(f"{'=' * 70}")
