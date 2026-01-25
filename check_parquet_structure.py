"""Check the structure of the parquet file"""
import pandas as pd

df = pd.read_parquet("data/airbnb_sampled_three_cities.parquet")
print(f"Total rows: {len(df)}")
print(f"\nAll columns ({len(df.columns)}):")
for col in df.columns:
    print(f"  - {col}")

# Check if there's review data
print("\n\nLooking for review-related columns:")
review_cols = [col for col in df.columns if 'review' in col.lower() or 'comment' in col.lower()]
print(f"Found: {review_cols}")

# Show sample row for property 40458495
print("\n\nSample row for property 40458495:")
row = df[df['property_id'] == 40458495].iloc[0]
for col in df.columns[:30]:  # First 30 columns
    print(f"  {col}: {row.get(col)}")
