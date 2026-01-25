"""
Manually ingest single property ABB_40458495 to airbnb-index
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from pyspark.sql import SparkSession
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone

# Configuration
PROPERTY_ID = "40458495"  # Raw ID
TARGET_PROPERTY = f"ABB_{PROPERTY_ID}"
INDEX_NAME = "airbnb-index"
DATA_FILE = "data/airbnb_sampled_three_cities.parquet"

print("=" * 70)
print(f"MANUAL INGESTION: {TARGET_PROPERTY}")
print("=" * 70)
print(f"Target index: {INDEX_NAME}")
print(f"Data file: {DATA_FILE}")
print("=" * 70)

# Create Spark session
spark = SparkSession.builder \
    .appName("Manual Property Ingestion") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load data
print("\n[1/5] Loading data...")
df = spark.read.parquet(DATA_FILE)
print(f"Total rows in file: {df.count()}")

# Find the specific property
print(f"\n[2/5] Finding property {PROPERTY_ID}...")
property_df = df.filter(df.id == PROPERTY_ID)

if property_df.count() == 0:
    print(f"[ERROR] Property {PROPERTY_ID} not found!")
    spark.stop()
    sys.exit(1)

prop = property_df.first()
print(f"[OK] Property found:")
print(f"  Name: {prop['name']}")
print(f"  City: {prop['city']}")
print(f"  Rating: {prop['rating']}")

# Create document
print(f"\n[3/5] Creating document...")

# Build property description
text_parts = [
    f"Property ID: {TARGET_PROPERTY}",
    f"Name: {prop['name']}",
    f"City: {prop['city']}, {prop['state']}, {prop['country']}",
    f"Property Type: {prop.get('property_type', 'N/A')}",
    f"Room Type: {prop.get('room_type', 'N/A')}",
    f"Rating: {prop['rating']}",
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

metadata = {
    "hotel_id": TARGET_PROPERTY,
    "title": prop['name'],
    "city": prop['city'],
    "state": prop.get('state', ''),
    "country": prop.get('country', ''),
    "rating": float(prop['rating']) if prop['rating'] else 0.0,
    "price": float(prop['price']) if prop.get('price') else 0.0,
    "property_type": prop.get('property_type', ''),
    "room_type": prop.get('room_type', ''),
}

document = Document(page_content=page_content, metadata=metadata)

print(f"[OK] Document created")
print(f"  Content length: {len(page_content)} chars")
print(f"  Metadata keys: {list(metadata.keys())}")

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
    spark.stop()
    sys.exit(1)

spark.stop()

print("\n" + "=" * 70)
print("[SUCCESS] Property ingested successfully!")
print("=" * 70)
print(f"You can now search for reviews of {TARGET_PROPERTY}")
print("=" * 70)
