"""Check what Airbnb properties are actually in the database"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

INDEX_NAME = "airbnb-index"  # Updated to new index

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')

print("\n" + "="*70)
print("AIRBNB PROPERTIES IN DATABASE")
print("="*70)

try:
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings, 
        namespace='airbnb_hotels'
    )
    
    # Get sample properties
    results = vectorstore.similarity_search("property Broadbeach", k=20)
    
    if results:
        print(f"\nFound {len(results)} Airbnb properties (showing first 20):\n")
        
        broadbeach_props = []
        other_props = []
        
        for doc in results:
            hotel_id = doc.metadata.get('hotel_id', 'N/A')
            title = doc.metadata.get('title', 'N/A')
            city = doc.metadata.get('city', 'N/A')
            rating = doc.metadata.get('rating', 'N/A')
            
            prop_info = f"  {hotel_id}: {title[:50]} | {city} | Rating: {rating}"
            
            if 'broadbeach' in city.lower():
                broadbeach_props.append(prop_info)
            else:
                other_props.append(prop_info)
        
        if broadbeach_props:
            print(f"Broadbeach Properties ({len(broadbeach_props)}):")
            for p in broadbeach_props:
                print(p)
        
        if other_props:
            print(f"\nOther Cities ({len(other_props)}):")
            for p in other_props[:10]:  # Show first 10
                print(p)
    else:
        print("No Airbnb properties found in database!")
        
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*70)
