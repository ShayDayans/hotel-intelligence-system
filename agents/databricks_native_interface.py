"""
Native Databricks Interface - No External Dependencies

This uses built-in Databricks widgets and displayHTML.
Most reliable option for long-running jobs on Databricks.

Copy this entire code to a Databricks notebook cell and run it.
"""

import sys
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system")
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system/agents")

import builtins
builtins.dbutils = dbutils

import time
from datetime import datetime
from agents.coordinator import LangGraphCoordinator

# ==============================================
# CONFIGURATION
# ==============================================

HOTEL_ID = "ABB_40458495"
HOTEL_NAME = "Rental unit in Broadbeach"
CITY = "Broadbeach"

# ==============================================
# INITIALIZE
# ==============================================

print("="*60)
print(f"🏨 HOTEL INTELLIGENCE SYSTEM")
print(f"Hotel: {HOTEL_NAME}")
print(f"City: {CITY}")
print("="*60)

coordinator = LangGraphCoordinator(HOTEL_ID, HOTEL_NAME, CITY)
state = coordinator.get_initial_state()

# ==============================================
# CREATE INPUT WIDGET
# ==============================================

dbutils.widgets.removeAll()

# Dropdown with example queries
dbutils.widgets.dropdown(
    name="query_type",
    defaultValue="Custom Query",
    choices=[
        "Custom Query",
        "What are guests saying about wifi?",
        "How do I compare to competitors?",
        "Am I more responsive than my competitors?",
        "What features should I improve?"
    ],
    label="Select Query Type"
)

# Text input for custom query
dbutils.widgets.text(
    name="custom_query",
    defaultValue="",
    label="Custom Query (if 'Custom Query' selected above)"
)

# ==============================================
# GET QUERY
# ==============================================

query_type = dbutils.widgets.get("query_type")
custom_query = dbutils.widgets.get("custom_query")

if query_type == "Custom Query":
    query = custom_query.strip()
    if not query:
        displayHTML("""
        <div style="padding: 20px; background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 5px;">
            <h3>⚠️ Please enter a custom query</h3>
            <p>Fill in the "custom_query" widget above and re-run this cell.</p>
        </div>
        """)
        raise ValueError("No query provided")
else:
    query = query_type

# ==============================================
# DISPLAY QUERY INFO
# ==============================================

displayHTML(f"""
<div style="padding: 20px; background-color: #e3f2fd; border-left: 5px solid #2196F3; margin-bottom: 20px;">
    <h2>🔍 Query</h2>
    <p style="font-size: 16px; margin: 10px 0;"><strong>{query}</strong></p>
    <p style="color: #666; font-size: 14px;">Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</div>
""")

# ==============================================
# RUN ANALYSIS
# ==============================================

print("\n" + "="*60)
print("🚀 STARTING ANALYSIS")
print("="*60)
print(f"Query: {query}")
print(f"This may take 1-20 minutes depending on complexity...")
print("="*60 + "\n")

start_time = time.time()
checkpoint_times = []

try:
    # Checkpoint 1
    print("✓ Phase 1: Extracting entities...")
    checkpoint_times.append(("Entity Extraction", time.time() - start_time))
    
    # Checkpoint 2
    print("✓ Phase 2: Routing to specialist agent(s)...")
    checkpoint_times.append(("Routing", time.time() - start_time))
    
    # Checkpoint 3 - Main execution
    print("✓ Phase 3: Running agent analysis...")
    print("  (This is the long-running part - please wait)\n")
    
    phase3_start = time.time()
    response, state = coordinator.run(query, state)
    phase3_time = time.time() - phase3_start
    
    checkpoint_times.append(("Agent Execution", phase3_time))
    
    total_time = time.time() - start_time
    mins, secs = divmod(int(total_time), 60)
    
    # Display success
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Total time: {mins}m {secs}s")
    print("\nTiming breakdown:")
    for phase, duration in checkpoint_times:
        print(f"  - {phase}: {duration:.1f}s")
    print("="*60 + "\n")
    
    # Display results with HTML formatting
    displayHTML(f"""
    <div style="padding: 20px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin-top: 20px;">
        <h2>✅ Analysis Complete!</h2>
        <p style="color: #666; font-size: 14px;">
            Completed in {mins}m {secs}s | 
            Ended: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </p>
    </div>
    
    <div style="padding: 20px; background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 5px; margin-top: 20px;">
        <h2>📊 Results</h2>
        <div style="line-height: 1.6; white-space: pre-wrap; font-family: Arial, sans-serif;">
{response}
        </div>
    </div>
    
    <div style="padding: 15px; background-color: #fff; border: 1px solid #ddd; border-radius: 5px; margin-top: 20px;">
        <h3>⏱️ Performance Breakdown</h3>
        <ul>
            {"".join([f"<li><strong>{phase}:</strong> {duration:.1f}s</li>" for phase, duration in checkpoint_times])}
        </ul>
        <p><strong>Total:</strong> {mins}m {secs}s</p>
    </div>
    
    <div style="padding: 15px; background-color: #e1f5fe; border-radius: 5px; margin-top: 20px;">
        <h3>🔄 Run Another Query</h3>
        <p>Change the widgets above and re-run this cell to ask another question.</p>
    </div>
    """)
    
    # Also print the response as text
    print("\n" + "="*60)
    print("RESPONSE:")
    print("="*60)
    print(response)
    print("="*60 + "\n")
    
except Exception as e:
    import traceback
    error_trace = traceback.format_exc()
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    
    print("\n" + "!"*60)
    print("❌ ERROR")
    print("!"*60)
    print(f"Failed after {mins}m {secs}s")
    print(f"\nError: {str(e)}")
    print("\nFull traceback:")
    print(error_trace)
    print("!"*60 + "\n")
    
    displayHTML(f"""
    <div style="padding: 20px; background-color: #ffebee; border-left: 5px solid #f44336; margin-top: 20px;">
        <h2>❌ Error Occurred</h2>
        <p style="color: #666; font-size: 14px;">Failed after {mins}m {secs}s</p>
        <p style="font-weight: bold; color: #c62828; margin: 15px 0;">{str(e)}</p>
        <details>
            <summary style="cursor: pointer; color: #1976d2;">Show full error trace</summary>
            <pre style="background-color: #fff; padding: 15px; border-radius: 5px; overflow-x: auto; margin-top: 10px;">{error_trace}</pre>
        </details>
    </div>
    """)

# ==============================================
# CONVERSATION HISTORY
# ==============================================

turn_count = state.get("turn_count", 0)
recent_turns = state.get("recent_turns", [])

if recent_turns:
    print("\n" + "="*60)
    print(f"CONVERSATION HISTORY ({turn_count} turns)")
    print("="*60)
    for i, turn in enumerate(recent_turns[-5:], 1):  # Show last 5 turns
        role = turn.get("role", "unknown")
        content = turn.get("content", "")[:200]  # Truncate for display
        print(f"\n{i}. [{role.upper()}] {content}...")
    print("\n" + "="*60)
