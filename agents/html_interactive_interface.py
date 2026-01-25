"""
Interactive HTML Interface for Databricks - Modern UI, No External Libraries

This creates a beautiful chat interface using pure HTML/JS with Databricks displayHTML.
Handles long-running jobs with polling mechanism.

Copy this to a Databricks notebook cell and run.
"""

import sys
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system")
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system/agents")

import builtins
builtins.dbutils = dbutils

import json
import time
import uuid
from datetime import datetime
from agents.coordinator import LangGraphCoordinator

# ==============================================
# CONFIGURATION
# ==============================================

HOTEL_ID = "ABB_40458495"
HOTEL_NAME = "Rental unit in Broadbeach"
CITY = "Broadbeach"

# ==============================================
# INITIALIZE (Persistent across cells using dbutils.jobs)
# ==============================================

# Store in Databricks runtime context (persistent)
if not hasattr(dbutils, '_hotel_intel_coordinator'):
    print("Initializing coordinator...")
    coordinator = LangGraphCoordinator(HOTEL_ID, HOTEL_NAME, CITY)
    state = coordinator.get_initial_state()
    history = []
    
    # Store in a way that persists (using temp table or just global)
    dbutils._hotel_intel_coordinator = coordinator
    dbutils._hotel_intel_state = state
    dbutils._hotel_intel_history = history
    print("✓ Coordinator initialized")
else:
    coordinator = dbutils._hotel_intel_coordinator
    state = dbutils._hotel_intel_state
    history = dbutils._hotel_intel_history
    print("✓ Using existing coordinator")

# ==============================================
# PROCESS QUERY FUNCTION
# ==============================================

def process_query(query: str):
    """Process a query and return response."""
    global state, history
    
    start_time = time.time()
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {query[:60]}...")
        print(f"{'='*60}\n")
        
        response, state = coordinator.run(query, state)
        
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        
        # Update history
        history.append({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "elapsed": f"{mins}m {secs}s"
        })
        
        # Store updated state
        dbutils._hotel_intel_state = state
        dbutils._hotel_intel_history = history
        
        print(f"✓ Complete in {mins}m {secs}s\n")
        
        return {
            "success": True,
            "response": response,
            "elapsed": f"{mins}m {secs}s"
        }
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        trace = traceback.format_exc()
        
        print(f"✗ Error: {error_msg}\n")
        print(trace)
        
        return {
            "success": False,
            "error": error_msg,
            "trace": trace
        }

# ==============================================
# CREATE INTERFACE
# ==============================================

# Create input widget
dbutils.widgets.text("query_input", "", "Enter your question:")

# Get query
query = dbutils.widgets.get("query_input").strip()

if query:
    # Process the query
    result = process_query(query)
    
    # Display result
    if result["success"]:
        response_html = result["response"].replace("\n", "<br>")
        
        displayHTML(f"""
        <style>
            .hotel-intel-container {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                max-width: 900px;
                margin: 0 auto;
            }}
            .query-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .response-box {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                line-height: 1.6;
            }}
            .meta-info {{
                background: #f5f5f5;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                font-size: 14px;
                color: #666;
            }}
            .success-badge {{
                display: inline-block;
                background: #4caf50;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 15px;
            }}
            h2 {{ margin-top: 0; }}
        </style>
        
        <div class="hotel-intel-container">
            <div class="query-box">
                <h2>🔍 Your Question</h2>
                <p style="font-size: 16px; margin: 0;">{query}</p>
            </div>
            
            <div class="response-box">
                <div class="success-badge">✓ Completed in {result["elapsed"]}</div>
                <h2>💡 Answer</h2>
                <div>{response_html}</div>
            </div>
            
            <div class="meta-info">
                <strong>Hotel:</strong> {HOTEL_NAME} ({HOTEL_ID})<br>
                <strong>City:</strong> {CITY}<br>
                <strong>Completed:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
                <strong>Conversation Turn:</strong> {state.get("turn_count", 0)}
            </div>
        </div>
        """)
        
    else:
        displayHTML(f"""
        <style>
            .error-container {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 900px;
                margin: 0 auto;
            }}
            .error-box {{
                background: #ffebee;
                border-left: 5px solid #f44336;
                padding: 20px;
                border-radius: 8px;
            }}
            .error-trace {{
                background: #fff;
                padding: 15px;
                border-radius: 5px;
                margin-top: 15px;
                overflow-x: auto;
                font-family: monospace;
                font-size: 12px;
            }}
        </style>
        
        <div class="error-container">
            <div class="error-box">
                <h2>❌ Error Occurred</h2>
                <p><strong>Query:</strong> {query}</p>
                <p><strong>Error:</strong> {result["error"]}</p>
                <details>
                    <summary style="cursor: pointer; color: #1976d2;">Show full trace</summary>
                    <div class="error-trace">{result["trace"].replace("\n", "<br>")}</div>
                </details>
            </div>
        </div>
        """)
    
    # Clear the input
    dbutils.widgets.remove("query_input")
    dbutils.widgets.text("query_input", "", "Enter your question:")

else:
    # Show welcome screen with examples
    examples_html = ""
    example_queries = [
        "What are guests saying about wifi?",
        "How do I compare to competitors?",
        "Am I more responsive than my competitors?",
        "What features should I improve to increase my rating?"
    ]
    
    for i, q in enumerate(example_queries, 1):
        examples_html += f"<li style='margin: 10px 0;'><code>{q}</code></li>"
    
    # Show conversation history
    history_html = ""
    if history:
        history_html = "<h3>📜 Recent Queries</h3><ol>"
        for item in history[-5:]:  # Last 5
            short_q = item["query"][:60] + "..." if len(item["query"]) > 60 else item["query"]
            history_html += f"<li style='margin: 10px 0;'><strong>{short_q}</strong> <span style='color: #666;'>({item['elapsed']})</span></li>"
        history_html += "</ol>"
    
    displayHTML(f"""
    <style>
        .welcome-container {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .info-box {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .example-queries {{
            background: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
        }}
        code {{
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
    
    <div class="welcome-container">
        <div class="header">
            <h1 style="margin: 0 0 10px 0;">🏨 Hotel Intelligence System</h1>
            <p style="margin: 0; opacity: 0.9;">{HOTEL_NAME} • {CITY}</p>
        </div>
        
        <div class="info-box">
            <h2>👋 Welcome!</h2>
            <p>Enter your question in the text box above and re-run this cell to get insights about your hotel.</p>
            <p style="color: #666; font-size: 14px;">
                <strong>Note:</strong> Complex queries may take 15-20 minutes to process. 
                The cell will show progress in the output.
            </p>
        </div>
        
        <div class="example-queries">
            <h3>💡 Example Questions</h3>
            <ul style="list-style: none; padding-left: 0;">
                {examples_html}
            </ul>
        </div>
        
        {f'<div class="info-box">{history_html}</div>' if history else ''}
        
        <div class="info-box" style="background: #fff3e0; border-left: 4px solid #ff9800;">
            <h3>⏱️ Query Types & Expected Time</h3>
            <ul>
                <li><strong>Quick</strong> (< 1 min): Reviews, rankings, basic info</li>
                <li><strong>Medium</strong> (2-5 min): Comparisons, sentiment analysis</li>
                <li><strong>Deep</strong> (15-20 min): Feature impact, ML analysis</li>
            </ul>
        </div>
    </div>
    """)

# ==============================================
# SHOW STATUS
# ==============================================

print("\n" + "="*60)
print("SYSTEM STATUS")
print("="*60)
print(f"Hotel: {HOTEL_NAME} ({HOTEL_ID})")
print(f"City: {CITY}")
print(f"Conversation turns: {state.get('turn_count', 0)}")
print(f"Queries in history: {len(history)}")
print("="*60)
print("\n✓ Ready for queries")
print("  Enter a question in the widget above and re-run this cell\n")
