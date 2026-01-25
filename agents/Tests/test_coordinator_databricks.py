"""
Test coordinator directly on Databricks (without Gradio)
Run this in a separate cell to verify the coordinator works
"""

import sys
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system")
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system/agents")

import builtins
builtins.dbutils = dbutils

import time
from agents.coordinator import LangGraphCoordinator

# ==============================================
# CONFIGURATION
# ==============================================

HOTEL_ID = "ABB_40458495"
HOTEL_NAME = "Rental unit in Broadbeach"
CITY = "Broadbeach"

print("="*60)
print("DIRECT COORDINATOR TEST (No Gradio)")
print("="*60)

# Test query
query = "Am I more responsive to my guests compared to my competitors?"

print(f"\nQuery: {query}")
print("\nInitializing coordinator...")

coordinator = LangGraphCoordinator(HOTEL_ID, HOTEL_NAME, CITY)
state = coordinator.get_initial_state()

print(f"✓ Coordinator initialized")
print(f"Initial state turn_count: {state.get('turn_count', 0)}")

print(f"\n{'='*60}")
print("Starting analysis...")
print(f"{'='*60}\n")

start_time = time.time()

try:
    response, state = coordinator.run(query, state)
    
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    
    print(f"\n{'='*60}")
    print(f"✓ SUCCESS!")
    print(f"{'='*60}")
    print(f"Time: {mins}m {secs}s")
    print(f"Response length: {len(response)} chars")
    print(f"Updated state turn_count: {state.get('turn_count', 0)}")
    print(f"\n--- RESPONSE ---")
    print(response)
    print(f"\n{'='*60}")
    
except Exception as e:
    import traceback
    elapsed = time.time() - start_time
    
    print(f"\n{'!'*60}")
    print(f"✗ ERROR after {elapsed:.1f}s")
    print(f"{'!'*60}")
    print(f"Exception: {e}")
    print(f"\nFull traceback:")
    traceback.print_exc()
    print(f"{'!'*60}")
