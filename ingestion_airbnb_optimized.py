"""
Memory-Optimized Airbnb Ingestion Pipeline

Fixes for common issues:
1. Out of memory (OOM) - Process 1 document at a time with garbage collection
2. Large embedding model - Uses truncated text and smaller batch sizes
3. Progress tracking - Saves progress to resume after failures
4. Network issues - Retries with exponential backoff

Usage:
    python ingestion_airbnb_optimized.py
"""

import os
import sys
import gc
import time
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Fix for PySpark on Windows
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Reduce PyTorch memory usage
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# ===========================================
# CONFIGURATION
# ===========================================

AIRBNB_PREFIX = "ABB"
PROGRESS_FILE = "ingestion_progress.json"

# Memory-optimized settings
BATCH_SIZE = 1  # Process 1 document at a time to avoid OOM
MAX_TEXT_LENGTH = 2000  # Truncate long texts to save memory
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds


def get_spark_session():
    """Initialize Spark session with minimal memory."""
    return SparkSession.builder \
        .appName("AirbnbIngestion") \
        .master("local[1]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.driver.memory", "512m") \
        .config("spark.executor.memory", "512m") \
        .config("spark.driver.maxResultSize", "256m") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .getOrCreate()


def create_pinecone_index_if_not_exists(index_name: str, dimension: int = 1024):
    """Create Pinecone index if it doesn't exist."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Waiting for index to initialize...")
        time.sleep(30)
    else:
        print(f"Index '{index_name}' already exists.")


def load_progress(progress_file: str) -> dict:
    """Load progress from file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"last_index": -1, "uploaded_ids": []}


def save_progress(progress_file: str, last_index: int, uploaded_ids: list):
    """Save progress to file."""
    with open(progress_file, 'w') as f:
        json.dump({"last_index": last_index, "uploaded_ids": uploaded_ids}, f)


def clear_memory():
    """Force garbage collection to free memory."""
    gc.collect()
    # Give system time to release memory
    time.sleep(0.1)


def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Truncate text to max length to save memory during embedding."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def process_airbnb_row(row) -> Document:
    """Process a single Airbnb row into a Document."""
    original_id = row['property_id'] or str(hash(row['name'] or ""))[:10]
    hotel_id = f"{AIRBNB_PREFIX}_{original_id}"
    
    # Combine name fields
    title = row['listing_name'] or row['name'] or "Airbnb Property"
    
    # Parse rating safely
    rating = 0.0
    if row['ratings']:
        try:
            rating = float(str(row['ratings']).replace(',', '.'))
        except (ValueError, AttributeError):
            pass
    
    # Build content with truncation
    description = truncate_text(row['description'] or "", 1000)
    amenities = truncate_text(row['amenities'] or "", 500)
    
    content = (
        f"Property: {title}. "
        f"Location: {row['location'] or 'Unknown'}, {row['country'] or 'Unknown'}. "
        f"Rating: {rating}. "
        f"Guests: {row['guests'] or 'N/A'}. "
        f"Category: {row['category'] or 'N/A'}. "
        f"Price: {row['price'] or 'N/A'}. "
        f"Amenities: {amenities}. "
        f"Description: {description}"
    )
    
    # Truncate total content
    content = truncate_text(content, MAX_TEXT_LENGTH)
    
    metadata = {
        "source": "airbnb",
        "hotel_id": hotel_id,
        "original_id": str(original_id),
        "title": title[:200],  # Limit metadata string length
        "city": (row['location'] or "")[:100],
        "country": (row['country'] or "")[:50],
        "rating": rating,
    }
    
    return Document(page_content=content, metadata=metadata)


def embed_with_retry(embeddings, text: str, attempts: int = RETRY_ATTEMPTS) -> list:
    """Embed text with retry logic."""
    for attempt in range(attempts):
        try:
            # Clear memory before embedding
            clear_memory()
            
            # Embed single document
            result = embeddings.embed_documents([text])
            return result[0]
        except Exception as e:
            print(f"\n    [Attempt {attempt + 1}/{attempts}] Embedding error: {e}")
            if attempt < attempts - 1:
                clear_memory()
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise
    return None


def upsert_with_retry(index, vectors: list, namespace: str, attempts: int = RETRY_ATTEMPTS):
    """Upsert to Pinecone with retry logic."""
    for attempt in range(attempts):
        try:
            index.upsert(vectors=vectors, namespace=namespace)
            return True
        except Exception as e:
            print(f"\n    [Attempt {attempt + 1}/{attempts}] Upsert error: {e}")
            if attempt < attempts - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise
    return False


def upload_airbnb_optimized(
    airbnb_path: str = "data/airbnb_sampled_three_cities.parquet",
    index_name: str = "airbnb-index",
    sample_size: int = 1000,
    city_filter: str = None,
    resume: bool = True
):
    """
    Memory-optimized Airbnb upload.
    
    Processes one document at a time with explicit garbage collection
    to avoid out-of-memory errors.
    
    Args:
        airbnb_path: Path to Airbnb parquet file
        index_name: Pinecone index name
        sample_size: Number of properties to upload
        city_filter: If set, only include properties from this city
        resume: If True, resume from last progress
    """
    print("=" * 60)
    print("MEMORY-OPTIMIZED AIRBNB INGESTION")
    print("=" * 60)
    print(f"Index: {index_name}")
    print(f"Source: {airbnb_path}")
    print(f"Sample Size: {sample_size}")
    print(f"Batch Size: {BATCH_SIZE} (memory-safe)")
    print(f"Max Text Length: {MAX_TEXT_LENGTH} chars")
    if city_filter:
        print(f"City Filter: {city_filter}")
    print("=" * 60)
    
    # Load progress
    progress = load_progress(PROGRESS_FILE) if resume else {"last_index": -1, "uploaded_ids": []}
    start_index = progress["last_index"] + 1
    uploaded_ids = set(progress["uploaded_ids"])
    
    if start_index > 0:
        print(f"\n[RESUME] Continuing from index {start_index} ({len(uploaded_ids)} already uploaded)")
    
    # Create Pinecone index
    create_pinecone_index_if_not_exists(index_name, dimension=1024)
    
    # Initialize Spark with minimal memory
    print("\n[SPARK] Initializing (minimal memory mode)...")
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    
    # Load data
    if not os.path.exists(airbnb_path):
        print(f"\n[ERROR] File not found: {airbnb_path}")
        spark.stop()
        return
    
    print(f"\n[LOAD] Reading parquet file...")
    df = spark.read.parquet(airbnb_path)
    
    # Select only needed columns to save memory
    df = df.select(
        col("property_id"),
        col("name"),
        col("listing_name"),
        col("description"),
        col("location"),
        col("country"),
        col("ratings"),
        col("amenities"),
        col("guests"),
        col("price"),
        col("category")
    ).fillna({
        "description": "",
        "name": "Unknown Property",
        "location": "Unknown",
        "country": "Unknown"
    })
    
    # Apply city filter if specified
    if city_filter:
        df = df.filter(col("location").contains(city_filter))
        print(f"[FILTER] Found properties in {city_filter}")
    
    # Limit sample size
    df = df.limit(sample_size)
    
    # Collect to driver (this is needed for iteration)
    print(f"[COLLECT] Loading {sample_size} rows into memory...")
    rows = df.collect()
    total_rows = len(rows)
    print(f"[COLLECT] Loaded {total_rows} rows")
    
    # Stop Spark early to free memory
    spark.stop()
    print("[SPARK] Stopped Spark to free memory")
    clear_memory()
    
    # Initialize embeddings model AFTER stopping Spark
    print("\n[MODEL] Loading embedding model (BAAI/bge-m3)...")
    print("[MODEL] This may take a moment...")
    
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 1  # Force single-item batches
        }
    )
    print("[MODEL] Model loaded")
    clear_memory()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    
    # Process documents one at a time
    print("\n" + "=" * 60)
    print("UPLOADING TO PINECONE (One at a time)")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    for i, row in enumerate(rows):
        if i < start_index:
            continue  # Skip already processed
        
        try:
            # Process row into document
            doc = process_airbnb_row(row)
            
            # Skip if already uploaded
            if doc.metadata['hotel_id'] in uploaded_ids:
                continue
            
            # Progress indicator
            elapsed = time.time() - start_time
            rate = (i - start_index + 1) / max(elapsed, 1)
            eta = (total_rows - i - 1) / max(rate, 0.01)
            
            print(f"\r[{i+1}/{total_rows}] {doc.metadata['hotel_id'][:30]:30} | "
                  f"{rate:.1f}/s | ETA: {eta/60:.1f}min", end="", flush=True)
            
            # Embed with memory management
            embedding = embed_with_retry(embeddings, doc.page_content)
            
            if embedding is None:
                print(f"\n  [SKIP] Failed to embed {doc.metadata['hotel_id']}")
                error_count += 1
                continue
            
            # Prepare vector
            vector = {
                "id": doc.metadata['hotel_id'],
                "values": embedding,
                "metadata": doc.metadata
            }
            
            # Upsert to Pinecone
            upsert_with_retry(index, [vector], "airbnb_hotels")
            
            # Track progress
            success_count += 1
            uploaded_ids.add(doc.metadata['hotel_id'])
            
            # Save progress every 10 documents
            if success_count % 10 == 0:
                save_progress(PROGRESS_FILE, i, list(uploaded_ids))
            
            # Clear memory periodically
            if i % 5 == 0:
                clear_memory()
                
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Saving progress...")
            save_progress(PROGRESS_FILE, i - 1, list(uploaded_ids))
            print(f"[SAVED] Progress saved at index {i-1}")
            print(f"[RESUME] Run again to continue from where you left off")
            return
            
        except Exception as e:
            print(f"\n  [ERROR] Row {i}: {e}")
            error_count += 1
            clear_memory()
            continue
    
    # Final progress save
    save_progress(PROGRESS_FILE, total_rows - 1, list(uploaded_ids))
    
    # Summary
    elapsed_total = time.time() - start_time
    print("\n\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total Time: {elapsed_total/60:.1f} minutes")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Rate: {success_count/max(elapsed_total, 1):.2f} docs/sec")
    print(f"Index: {index_name}")
    print(f"Namespace: airbnb_hotels")
    print("=" * 60)
    
    # Clean up progress file on successful completion
    if error_count == 0 and success_count == total_rows:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            print("[CLEANUP] Removed progress file (complete)")


def clear_and_restart(
    airbnb_path: str = "data/airbnb_sampled_three_cities.parquet",
    index_name: str = "airbnb-index",
    sample_size: int = 1000
):
    """Clear existing data and start fresh."""
    print("[RESET] Clearing existing data...")
    
    # Remove progress file
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("[RESET] Removed progress file")
    
    # Clear Pinecone namespace
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(index_name)
        index.delete(delete_all=True, namespace="airbnb_hotels")
        print(f"[RESET] Cleared namespace 'airbnb_hotels' in index '{index_name}'")
        time.sleep(2)
    except Exception as e:
        print(f"[RESET] Could not clear Pinecone: {e}")
    
    # Run fresh ingestion
    upload_airbnb_optimized(
        airbnb_path=airbnb_path,
        index_name=index_name,
        sample_size=sample_size,
        resume=False
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory-optimized Airbnb ingestion")
    parser.add_argument("--path", default="data/airbnb_sampled_three_cities.parquet", 
                        help="Path to parquet file")
    parser.add_argument("--index", default="airbnb-index", 
                        help="Pinecone index name")
    parser.add_argument("--size", type=int, default=1000, 
                        help="Number of documents to upload")
    parser.add_argument("--city", default=None, 
                        help="Filter by city")
    parser.add_argument("--fresh", action="store_true", 
                        help="Clear existing data and start fresh")
    parser.add_argument("--no-resume", action="store_true", 
                        help="Don't resume from previous progress")
    
    args = parser.parse_args()
    
    if args.fresh:
        clear_and_restart(
            airbnb_path=args.path,
            index_name=args.index,
            sample_size=args.size
        )
    else:
        upload_airbnb_optimized(
            airbnb_path=args.path,
            index_name=args.index,
            sample_size=args.size,
            city_filter=args.city,
            resume=not args.no_resume
        )
