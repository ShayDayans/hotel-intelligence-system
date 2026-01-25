"""
Upload 1000 Airbnb properties to Pinecone
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion import run_ingestion

print("=" * 60)
print("UPLOADING 1000 AIRBNB PROPERTIES TO PINECONE")
print("=" * 60)

run_ingestion(
    booking_path="data/nonexistent.parquet",  # Skip booking data
    airbnb_path="data/airbnb_sampled_three_cities.parquet",
    index_name="booking-agent",
    sample_size=1000,  # Upload 1000 properties
    city_filter=None,  # Include all cities
    clear_existing=False  # Don't clear existing data, just add
)

print("\n" + "=" * 60)
print("UPLOAD COMPLETE!")
print("=" * 60)
