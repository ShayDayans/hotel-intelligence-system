"""
Check parquet for reviews and ingest them for property ABB_40458495
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone

# Configuration
PROPERTY_ID = "ABB_40458495"  # Full ID as stored in parquet
TARGET_PROPERTY = PROPERTY_ID
INDEX_NAME = "airbnb-index"
DATA_FILE = "data/airbnb_sampled_three_cities.parquet"

print("=" * 70)
print(f"CHECK AND INGEST REVIEWS FOR {TARGET_PROPERTY}")
print("=" * 70)

# Load data
print("\n[1/5] Loading parquet file...")
df = pd.read_parquet(DATA_FILE)
print(f"Total rows: {len(df)}")

# Check property_id column type
print(f"\nproperty_id dtype: {df['property_id'].dtype}")
print(f"Sample property_ids: {df['property_id'].head(5).tolist()}")

# Find the property - try different formats
print(f"\n[2/5] Finding property {PROPERTY_ID}...")

# Try exact match first
property_row = df[df['property_id'] == PROPERTY_ID]
if len(property_row) == 0:
    # Try without prefix
    property_row = df[df['property_id'] == "40458495"]
if len(property_row) == 0:
    # Try as int
    property_row = df[df['property_id'] == 40458495]
if len(property_row) == 0:
    # Try with contains
    property_row = df[df['property_id'].astype(str).str.contains("40458495")]

if len(property_row) == 0:
    print(f"[ERROR] Property {PROPERTY_ID} not found!")
    print("\nSearching for similar IDs...")
    # Check if there's any ID containing these digits
    sample = df['property_id'].head(20).tolist()
    print(f"First 20 property_ids: {sample}")
    sys.exit(1)

prop = property_row.iloc[0]
prop_name = str(prop.get('name', 'Unknown')).encode('ascii', 'replace').decode('ascii')
print(f"[OK] Found property: {prop_name}")

# Check review data
print(f"\n[3/5] Checking review data...")
reviews_data = prop.get('reviews')
print(f"reviews column value type: {type(reviews_data)}")

if pd.notna(reviews_data) and reviews_data:
    reviews_str = str(reviews_data)
    print(f"reviews length: {len(reviews_str)} chars")
    print(f"reviews preview: {reviews_str[:300]}...")
else:
    print("[WARNING] No reviews data for this property")
    sys.exit(0)


def parse_airbnb_reviews(reviews_str: str) -> list[str]:
    """Parse Airbnb reviews string into individual reviews."""
    if not reviews_str or str(reviews_str).strip() == "":
        return []

    reviews_str = str(reviews_str)
    
    # Try JSON parsing first
    try:
        parsed = json.loads(reviews_str)
        if isinstance(parsed, list):
            reviews = []
            for item in parsed:
                if isinstance(item, dict):
                    text = item.get('review') or item.get('text') or item.get('comment') or item.get('comments') or str(item)
                    reviews.append(text)
                elif isinstance(item, str):
                    reviews.append(item)
            return reviews
        elif isinstance(parsed, str):
            return [parsed]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: split by common delimiters
    import re
    if any(f"{i}." in reviews_str for i in range(1, 5)):
        parts = re.split(r'\d+\.\s*', reviews_str)
        return [p.strip() for p in parts if p.strip()]

    for delimiter in ['\n\n', '|', '---']:
        if delimiter in reviews_str:
            parts = reviews_str.split(delimiter)
            return [p.strip() for p in parts if p.strip()]

    return [reviews_str] if len(reviews_str) > 20 else []


# Parse reviews
print(f"\n[4/5] Parsing reviews...")
reviews = parse_airbnb_reviews(reviews_data)
print(f"[OK] Parsed {len(reviews)} review(s)")

if reviews:
    for i, rev in enumerate(reviews[:3]):  # Show first 3
        rev_preview = str(rev)[:100].encode('ascii', 'replace').decode('ascii')
        print(f"  Review {i+1}: {rev_preview}...")
    
    print(f"\n[5/5] Ingesting reviews to Pinecone...")
    
    # Load embeddings
    print("  Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')
    
    # Connect to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)
    
    # Create and upload review documents
    location = str(prop.get('location', 'Unknown')).encode('ascii', 'replace').decode('ascii')
    
    uploaded_count = 0
    for idx, review_text in enumerate(reviews):
        if not review_text or len(str(review_text)) < 10:
            continue
        
        review_id = f"{TARGET_PROPERTY}_R{idx}"
        content = f"Review for {prop_name} in {location}: {review_text}"
        
        # Clean content for embedding
        content_clean = content.encode('ascii', 'replace').decode('ascii')
        
        metadata = {
            "source": "airbnb",
            "review_id": review_id,
            "hotel_id": TARGET_PROPERTY,
            "hotel_name": prop_name,
            "city": location,
            "country": str(prop.get('country', '')).encode('ascii', 'replace').decode('ascii'),
        }
        
        # Generate embedding
        embedding = embeddings.embed_documents([content_clean])[0]
        
        # Upload to Pinecone
        vector = {
            "id": review_id,
            "values": embedding,
            "metadata": metadata
        }
        
        index.upsert(vectors=[vector], namespace="airbnb_reviews")
        uploaded_count += 1
        print(f"  [OK] Uploaded review {uploaded_count}: {review_id}")
    
    print(f"\n{'=' * 70}")
    print(f"[SUCCESS] Ingested {uploaded_count} reviews to airbnb_reviews namespace!")
    print(f"{'=' * 70}")
else:
    print("[WARNING] No parseable reviews found")
