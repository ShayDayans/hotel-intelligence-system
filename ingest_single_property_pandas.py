"""
Manually ingest single property ABB_40458495 to airbnb-index
(No Spark - uses pandas/pyarrow)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone

# Configuration
PROPERTY_ID = 40458495  # Raw ID as integer
TARGET_PROPERTY = f"ABB_{PROPERTY_ID}"
INDEX_NAME = "airbnb-index"
DATA_FILE = "data/airbnb_sampled_three_cities.parquet"

print("=" * 70)
print(f"MANUAL INGESTION: {TARGET_PROPERTY}")
print("=" * 70)
print(f"Target index: {INDEX_NAME}")
print(f"Data file: {DATA_FILE}")
print("=" * 70)

# Load data with pandas
print("\n[1/5] Loading data with pandas...")
df = pd.read_parquet(DATA_FILE)
print(f"Total rows in file: {len(df)}")
print(f"Columns: {list(df.columns[:10])}...")  # Show first 10 columns

# Find ID column
id_col = None
for col in df.columns:
    if 'id' in col.lower():
        print(f"  Found ID column candidate: {col}")
        id_col = col
        break

if id_col is None:
    print("Available columns:", list(df.columns))
    sys.exit(1)

# Find the specific property
print(f"\n[2/5] Finding property {PROPERTY_ID} in column '{id_col}'...")

# Try both string and int
property_row = df[df[id_col] == PROPERTY_ID]
if len(property_row) == 0:
    property_row = df[df[id_col] == str(PROPERTY_ID)]
if len(property_row) == 0:
    # Show sample values
    print(f"Sample values in {id_col}: {df[id_col].head(5).tolist()}")
    print(f"[ERROR] Property {PROPERTY_ID} not found!")
    sys.exit(1)

prop = property_row.iloc[0]
print(f"[OK] Property found:")
name_str = str(prop.get('name', 'N/A')).encode('ascii', 'replace').decode('ascii')
print(f"  Name: {name_str}")
print(f"  City: {prop.get('city', 'N/A')}")
print(f"  Rating: {prop.get('rating', 'N/A')}")

# Create document
print(f"\n[3/5] Creating document...")

# Build property description
text_parts = [
    f"Property ID: {TARGET_PROPERTY}",
    f"Name: {prop.get('name', 'N/A')}",
    f"City: {prop.get('city', 'N/A')}, {prop.get('state', 'N/A')}, {prop.get('country', 'N/A')}",
    f"Property Type: {prop.get('property_type', 'N/A')}",
    f"Room Type: {prop.get('room_type', 'N/A')}",
    f"Rating: {prop.get('rating', 'N/A')}",
    f"Number of Reviews: {prop.get('comments', 0)}",
    f"Price: ${prop.get('price', 'N/A')}",
    f"Accommodates: {prop.get('accommodates', 'N/A')} guests",
    f"Bedrooms: {prop.get('bedrooms', 'N/A')}",
    f"Beds: {prop.get('beds', 'N/A')}",
    f"Bathrooms: {prop.get('bathrooms', 'N/A')}",
]

# Add amenities if available
amenity_cols = [col for col in df.columns if col.startswith('amen_')]
amenities = []
for col in amenity_cols:
    if prop.get(col, 0) == 1:
        amenity_name = col.replace('amen_', '').replace('_', ' ').title()
        amenities.append(amenity_name)

if amenities:
    text_parts.append(f"Amenities: {', '.join(amenities)}")

page_content = "\n".join(text_parts)

def safe_float(val, default=0.0):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except:
        return default

def safe_str(val, default=''):
    if pd.isna(val):
        return default
    # Remove special Unicode characters that can cause encoding issues
    return str(val).encode('ascii', 'replace').decode('ascii')

metadata = {
    "hotel_id": TARGET_PROPERTY,
    "title": safe_str(prop.get('name')),
    "city": safe_str(prop.get('city')),
    "state": safe_str(prop.get('state')),
    "country": safe_str(prop.get('country')),
    "rating": safe_float(prop.get('rating')),
    "price": safe_float(prop.get('price')),
    "property_type": safe_str(prop.get('property_type')),
    "room_type": safe_str(prop.get('room_type')),
}

document = Document(page_content=page_content, metadata=metadata)

print(f"[OK] Document created")
print(f"  Content length: {len(page_content)} chars")
print(f"  Metadata: {metadata}")

# Load embeddings
print(f"\n[4/5] Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')
print(f"[OK] Embeddings loaded")

# Generate embedding
print(f"\n[5/5] Uploading to Pinecone...")
print(f"  Generating embedding...")
embedding = embeddings.embed_documents([document.page_content])[0]
print(f"[OK] Embedding generated (dimension: {len(embedding)})")

# Upload to Pinecone
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)
    
    vector = {
        "id": TARGET_PROPERTY,
        "values": embedding,
        "metadata": metadata
    }
    
    index.upsert(vectors=[vector], namespace="airbnb_hotels")
    print(f"[OK] Uploaded to Pinecone")
    print(f"     Index: {INDEX_NAME}")
    print(f"     Namespace: airbnb_hotels")
    print(f"     Vector ID: {TARGET_PROPERTY}")
    
except Exception as e:
    print(f"[ERROR] Upload failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("[SUCCESS] Property ingested successfully!")
print("=" * 70)
print(f"You can now search for {TARGET_PROPERTY}")
print("=" * 70)
