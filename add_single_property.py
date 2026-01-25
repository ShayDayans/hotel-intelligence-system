"""
Add a single property to Pinecone airbnb-index
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Fix for PySpark on Windows
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

# Target property
TARGET_ID = "40458495"  # Without ABB_ prefix
HOTEL_ID = f"ABB_{TARGET_ID}"
INDEX_NAME = "airbnb-index"
NAMESPACE = "airbnb_hotels"

def get_spark_session():
    return SparkSession.builder \
        .appName("SinglePropertyUpload") \
        .master("local[1]") \
        .config("spark.driver.memory", "512m") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

def main():
    print(f"=" * 60)
    print(f"ADDING SINGLE PROPERTY: {HOTEL_ID}")
    print(f"=" * 60)
    
    # Load data with Spark
    print("\n[1] Loading parquet file...")
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    
    df = spark.read.parquet("data/airbnb_sampled_three_cities.parquet")
    
    # Find the specific property
    print(f"[2] Searching for property_id = {TARGET_ID}...")
    
    # Try different ID formats
    property_row = df.filter(
        (col("property_id") == TARGET_ID) | 
        (col("property_id") == int(TARGET_ID) if TARGET_ID.isdigit() else col("property_id") == TARGET_ID)
    ).first()
    
    if not property_row:
        print(f"    Property not found with ID {TARGET_ID}")
        print("    Trying to list some property_ids...")
        sample = df.select("property_id", "name", "listing_name").limit(10).collect()
        for row in sample:
            print(f"      - {row['property_id']}: {row['listing_name'] or row['name']}")
        spark.stop()
        return
    
    print(f"    Found: {property_row['listing_name'] or property_row['name']}")
    
    # Extract data
    title = property_row['listing_name'] or property_row['name'] or "Airbnb Property"
    location = property_row['location'] or "Unknown"
    country = property_row['country'] or "Unknown"
    description = (property_row['description'] or "")[:1500]
    amenities = (property_row['amenities'] or "")[:500]
    guests = property_row['guests'] or "N/A"
    price = property_row['price'] or "N/A"
    category = property_row['category'] or "N/A"
    
    # Parse rating
    rating = 0.0
    if property_row['ratings']:
        try:
            rating = float(str(property_row['ratings']).replace(',', '.'))
        except:
            pass
    
    # Build content
    content = (
        f"Property: {title}. "
        f"Location: {location}, {country}. "
        f"Rating: {rating}. "
        f"Guests: {guests}. "
        f"Category: {category}. "
        f"Price: {price}. "
        f"Amenities: {amenities}. "
        f"Description: {description}"
    )[:2000]
    
    print(f"\n[3] Property details:")
    print(f"    Title: {title}")
    print(f"    Location: {location}, {country}")
    print(f"    Rating: {rating}")
    print(f"    Content length: {len(content)} chars")
    
    # Stop Spark to free memory
    spark.stop()
    print("\n[4] Spark stopped, loading embedding model...")
    
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 1}
    )
    print("    Model loaded")
    
    # Create embedding
    print("\n[5] Creating embedding...")
    embedding = embeddings.embed_documents([content])[0]
    print(f"    Embedding dimension: {len(embedding)}")
    
    # Prepare metadata
    metadata = {
        "source": "airbnb",
        "hotel_id": HOTEL_ID,
        "original_id": str(TARGET_ID),
        "title": title[:200],
        "city": location[:100],
        "country": country[:50],
        "rating": rating,
    }
    
    # Upload to Pinecone
    print(f"\n[6] Uploading to Pinecone...")
    print(f"    Index: {INDEX_NAME}")
    print(f"    Namespace: {NAMESPACE}")
    print(f"    Vector ID: {HOTEL_ID}")
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)
    
    vector = {
        "id": HOTEL_ID,
        "values": embedding,
        "metadata": metadata
    }
    
    index.upsert(vectors=[vector], namespace=NAMESPACE)
    
    print(f"\n{'=' * 60}")
    print(f"SUCCESS! Property {HOTEL_ID} added to Pinecone")
    print(f"{'=' * 60}")
    
    # Verify
    print("\n[7] Verifying upload...")
    result = index.fetch(ids=[HOTEL_ID], namespace=NAMESPACE)
    if result and result.vectors and HOTEL_ID in result.vectors:
        print(f"    ✓ Verified: {HOTEL_ID} exists in index")
        print(f"    ✓ Metadata: {result.vectors[HOTEL_ID].metadata}")
    else:
        print(f"    ✗ Could not verify upload")

if __name__ == "__main__":
    main()
