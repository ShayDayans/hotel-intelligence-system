"""
Count properties in Pinecone
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

def count_pinecone_properties():
    """Count total properties (hotels) in Pinecone."""
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("booking-agent")
    
    print("\n" + "="*60)
    print("PINECONE INDEX STATISTICS")
    print("="*60)
    
    # Get index stats
    stats = index.describe_index_stats()
    
    print(f"\nIndex: booking-agent")
    print(f"Total vectors: {stats.get('total_vector_count', 0):,}")
    print(f"Dimension: {stats.get('dimension', 'N/A')}")
    
    # Count by namespace
    namespaces = stats.get('namespaces', {})
    
    print("\n" + "-"*60)
    print("BREAKDOWN BY NAMESPACE:")
    print("-"*60)
    
    total_hotels = 0
    total_reviews = 0
    
    for namespace, ns_stats in namespaces.items():
        count = ns_stats.get('vector_count', 0)
        print(f"\n{namespace}:")
        print(f"  Vectors: {count:,}")
        
        if 'hotels' in namespace:
            total_hotels += count
            print(f"  (Properties)")
        elif 'reviews' in namespace:
            total_reviews += count
            print(f"  (Review documents)")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Properties (Hotels): {total_hotels:,}")
    print(f"Total Review Documents: {total_reviews:,}")
    print(f"Total Vectors: {stats.get('total_vector_count', 0):,}")
    print("="*60)
    
    return {
        "total_hotels": total_hotels,
        "total_reviews": total_reviews,
        "total_vectors": stats.get('total_vector_count', 0)
    }

if __name__ == "__main__":
    count_pinecone_properties()
