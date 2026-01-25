"""
Search for Cleanliness Reviews for Property ABB_40458495
"""

import sys
import os

# Add paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_THIS_DIR, '.env'))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Configuration
PROPERTY_ID = "ABB_40458495"
INDEX_NAME = "airbnb-index"  # Updated to new index
QUERY_TOPIC = "cleanliness"

print("=" * 70)
print(f"SEARCHING REVIEWS FOR PROPERTY {PROPERTY_ID}")
print("=" * 70)
print(f"Topic: {QUERY_TOPIC}")
print("=" * 70)

# Load embeddings
print("\nLoading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')
print("[OK] Embeddings loaded")

# Search Airbnb reviews namespace
print(f"\n[1] Searching airbnb_reviews namespace...")
try:
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings, 
        namespace='airbnb_reviews'
    )
    
    # Search with filter for specific property
    results = vectorstore.similarity_search(
        QUERY_TOPIC, 
        k=10,
        filter={"hotel_id": PROPERTY_ID}
    )
    
    if results:
        print(f"[OK] Found {len(results)} review(s) mentioning {QUERY_TOPIC}")
        print("-" * 70)
        
        for i, doc in enumerate(results, 1):
            print(f"\n--- Review {i} ---")
            print(f"Hotel ID: {doc.metadata.get('hotel_id', 'N/A')}")
            print(f"Hotel Name: {doc.metadata.get('hotel_name', 'N/A')}")
            print(f"City: {doc.metadata.get('city', 'N/A')}")
            print(f"Review ID: {doc.metadata.get('review_id', 'N/A')}")
            print(f"\nReview Text:")
            print(doc.page_content)
            print("-" * 70)
    else:
        print(f"[X] No reviews found for property {PROPERTY_ID} about {QUERY_TOPIC}")
        
        # Try searching without filter to see if property has ANY reviews
        print(f"\n[2] Checking if property has ANY reviews...")
        all_results = vectorstore.similarity_search(
            "review", 
            k=100,
            filter={"hotel_id": PROPERTY_ID}
        )
        
        if all_results:
            print(f"[OK] Found {len(all_results)} total review(s) for this property")
            print(f"   (But none specifically about {QUERY_TOPIC})")
            
            # Show first review as example
            if len(all_results) > 0:
                print(f"\n   Example review:")
                print(f"   {all_results[0].page_content[:200]}...")
        else:
            print(f"[X] Property {PROPERTY_ID} has NO reviews in the database")
            print(f"   This property may not have been ingested yet.")
            
except Exception as e:
    print(f"[X] Error searching reviews: {e}")

# Also check if the property itself exists
print(f"\n[3] Checking if property exists in airbnb_hotels namespace...")
try:
    vectorstore_hotels = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings, 
        namespace='airbnb_hotels'
    )
    
    hotel_results = vectorstore_hotels.similarity_search(
        "property", 
        k=10,
        filter={"hotel_id": PROPERTY_ID}
    )
    
    if hotel_results:
        print(f"[OK] Property exists in database")
        print(f"   Name: {hotel_results[0].metadata.get('title', 'N/A')}")
        print(f"   City: {hotel_results[0].metadata.get('city', 'N/A')}")
        print(f"   Rating: {hotel_results[0].metadata.get('rating', 'N/A')}")
    else:
        print(f"[X] Property {PROPERTY_ID} not found in airbnb_hotels")
        
except Exception as e:
    print(f"[X] Error checking property: {e}")

print(f"\n{'='*70}")
print("SEARCH COMPLETE")
print(f"{'='*70}")
