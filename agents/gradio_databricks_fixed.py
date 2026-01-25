"""
Gradio Chat Interface for Databricks - FIXED VERSION

Key fixes:
1. Thread-safe state handling (deep copy for each query)
2. Better error handling and logging
3. Timeout protection for long-running jobs
4. Progress updates that actually work
"""

import sys
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system")
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system/agents")

import builtins
builtins.dbutils = dbutils  # Required for Databricks

import gradio as gr
import base64
import threading
import time
import copy
from io import BytesIO
from PIL import Image as PILImage
from agents.coordinator import LangGraphCoordinator

# ==============================================
# CONFIGURATION
# ==============================================

HOTEL_ID = "ABB_40458495"
HOTEL_NAME = "Rental unit in Broadbeach"  
CITY = "Broadbeach"

print("[INIT] Creating coordinator...")
coordinator = LangGraphCoordinator(HOTEL_ID, HOTEL_NAME, CITY)
state = coordinator.get_initial_state()
print("[INIT] Coordinator ready")

# Thread-safe global storage
captured_charts = []
state_lock = threading.Lock()
job_lock = threading.Lock()

current_job = {
    "running": False,
    "progress": "",
    "result": None,
    "error": None,
    "start_time": None,
    "thread_id": None
}

# ==============================================
# CHART CAPTURE PATCH (with debug instrumentation)
# ==============================================

# #region agent log - DEBUG: track patch application
_patched_modules = []
print("[DEBUG-INIT] Chart capture patch section starting...")
# #endregion

def _capture_charts(result):
    global captured_charts
    # #region agent log - Hypothesis D: check chart structure
    print(f"[DEBUG-CAPTURE] _capture_charts called!")
    print(f"[DEBUG-CAPTURE] Result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
    # #endregion

    charts = result.get("charts", {})
    # #region agent log - Hypothesis D: check both chart locations
    print(f"[DEBUG-CAPTURE] Direct charts found: {bool(charts)}, keys: {list(charts.keys()) if charts else 'none'}")
    # #endregion
    
    if not charts:
        ui_artifacts = result.get("ui_artifacts", {})
        charts = ui_artifacts.get("charts", {}) if isinstance(ui_artifacts, dict) else {}
        # #region agent log - Hypothesis D: check ui_artifacts
        print(f"[DEBUG-CAPTURE] ui_artifacts.charts found: {bool(charts)}, keys: {list(charts.keys()) if charts else 'none'}")
        # #endregion

    captured_charts = []
    for name, b64 in charts.items():
        # #region agent log - Hypothesis D: check chart data format
        print(f"[DEBUG-CAPTURE] Processing chart '{name}', type: {type(b64)}, is_dict: {isinstance(b64, dict)}")
        # #endregion
        if isinstance(b64, dict):
            b64 = b64.get("data") or b64.get("base64") or b64.get("image")
            # #region agent log
            print(f"[DEBUG-CAPTURE] Extracted from dict, got data: {bool(b64)}")
            # #endregion
        if b64 and isinstance(b64, str):
            try:
                # #region agent log - Hypothesis E: check PIL decode
                print(f"[DEBUG-CAPTURE] Decoding base64 for '{name}', length: {len(b64)}")
                # #endregion
                img = PILImage.open(BytesIO(base64.b64decode(b64)))
                captured_charts.append(img)
                # #region agent log
                print(f"[DEBUG-CAPTURE] ✓ Successfully decoded '{name}', img size: {img.size}")
                # #endregion
            except Exception as e:
                print(f"[DEBUG-CAPTURE] ✗ Failed to decode {name}: {e}")
    
    # #region agent log - Hypothesis C: check final state
    print(f"[DEBUG-CAPTURE] Final captured_charts count: {len(captured_charts)}")
    print(f"[DEBUG-CAPTURE] captured_charts id: {id(captured_charts)}")
    # #endregion

def _patch_lr(module, module_name):
    original = module.run_lr_analysis
    # #region agent log - Hypothesis B: track which module is patched
    print(f"[DEBUG-PATCH] Patching {module_name}, original func id: {id(original)}")
    # #endregion

    def patched_lr(hotel_id, timeout_seconds=1200):
        # #region agent log - Hypothesis A/B: confirm patch is called
        print(f"[DEBUG-PATCH] ★★★ PATCHED run_lr_analysis CALLED! ★★★")
        print(f"[DEBUG-PATCH] Called from module: {module_name}")
        print(f"[DEBUG-PATCH] hotel_id: {hotel_id}")
        # #endregion
        result = original(hotel_id, timeout_seconds)
        # #region agent log
        print(f"[DEBUG-PATCH] Original returned, calling _capture_charts...")
        # #endregion
        _capture_charts(result)
        # #region agent log - Hypothesis C: check state after capture
        print(f"[DEBUG-PATCH] After _capture_charts, captured_charts count: {len(captured_charts)}")
        # #endregion
        return result

    module.run_lr_analysis = patched_lr
    _patched_modules.append(module_name)
    # #region agent log
    print(f"[DEBUG-PATCH] ✓ Patched {module_name}, new func id: {id(module.run_lr_analysis)}")
    # #endregion

try:
    import databricks_tools
    _patch_lr(databricks_tools, "databricks_tools")
    print("[OK] Chart capture enabled (databricks_tools)")
except Exception as e:
    print(f"[Note] Chart capture not available in databricks_tools: {e}")

try:
    from agents import databricks_tools as agents_databricks_tools
    _patch_lr(agents_databricks_tools, "agents.databricks_tools")
    print("[OK] Chart capture enabled (agents.databricks_tools)")
except Exception as e:
    print(f"[Note] Chart capture not available in agents.databricks_tools: {e}")

# #region agent log - show all patched modules
print(f"[DEBUG-INIT] Patched modules: {_patched_modules}")
# #endregion

# ==============================================
# ASYNC JOB HANDLING - FIXED VERSION
# ==============================================

def run_analysis_thread(query: str, thread_state: dict):
    """
    Run analysis in background thread with proper state isolation.
    
    Args:
        query: User query
        thread_state: Deep copy of current state for this thread
    """
    global captured_charts, current_job
    
    print(f"\n{'='*60}")
    print(f"[THREAD START] Query: {query[:80]}")
    print(f"[THREAD] Thread ID: {threading.current_thread().ident}")
    print(f"[THREAD] State turn_count: {thread_state.get('turn_count', 0)}")
    print(f"{'='*60}\n")
    
    # Update job status
    with job_lock:
        current_job["running"] = True
        current_job["progress"] = "Initializing..."
        current_job["result"] = None
        current_job["error"] = None
        current_job["start_time"] = time.time()
        current_job["thread_id"] = threading.current_thread().ident
        captured_charts = []
    
    result_state = None
    
    try:
        # Step 1: Entity extraction
        with job_lock:
            current_job["progress"] = "Extracting entities from query..."
        print("[THREAD] Phase 1: Entity extraction")
        time.sleep(0.5)  # Give status time to update
        
        # Step 2: Routing
        with job_lock:
            current_job["progress"] = "Routing to specialist agent(s)..."
        print("[THREAD] Phase 2: Routing")
        time.sleep(0.5)
        
        # Step 3: Execute coordinator
        with job_lock:
            current_job["progress"] = "Agent working (may take 15-20 min for deep analysis)..."
        
        print(f"[THREAD] Phase 3: Calling coordinator.run()")
        print(f"[THREAD] Query length: {len(query)} chars")
        
        # THE ACTUAL CALL - with timeout protection
        start_exec = time.time()
        
        # Run the coordinator (this is the long-running part)
        response, result_state = coordinator.run(query, thread_state)
        
        exec_time = time.time() - start_exec
        
        print(f"[THREAD] ✓ Coordinator completed!")
        print(f"[THREAD] Execution time: {exec_time:.1f}s ({exec_time/60:.1f} min)")
        print(f"[THREAD] Response length: {len(response)} chars")
        print(f"[THREAD] Response preview: {response[:200]}...")
        # #region agent log - Hypothesis C: check captured_charts after coordinator
        print(f"[DEBUG-THREAD] After coordinator.run, captured_charts count: {len(captured_charts)}")
        print(f"[DEBUG-THREAD] captured_charts id: {id(captured_charts)}")
        print(f"[DEBUG-THREAD] captured_charts types: {[type(c).__name__ for c in captured_charts]}")
        # #endregion
        
        # Step 4: Success
        with job_lock:
            current_job["result"] = response
            current_job["progress"] = f"Complete! (took {exec_time/60:.1f} min)"
        
        # Update global state with result
        with state_lock:
            global state
            state = result_state
            print(f"[THREAD] Global state updated. New turn_count: {state.get('turn_count', 0)}")
            
    except Exception as e:
        # Detailed error logging
        import traceback
        error_details = traceback.format_exc()
        
        print(f"\n{'!'*60}")
        print(f"[THREAD ERROR] Exception occurred!")
        print(f"[THREAD ERROR] Type: {type(e).__name__}")
        print(f"[THREAD ERROR] Message: {str(e)}")
        print(f"[THREAD ERROR] Traceback:\n{error_details}")
        print(f"{'!'*60}\n")
        
        with job_lock:
            current_job["error"] = str(e)
            current_job["progress"] = f"Error: {str(e)[:100]}"
    
    finally:
        print(f"\n[THREAD END] Cleaning up")
        print(f"[THREAD] Final status - Result: {bool(current_job.get('result'))}, Error: {bool(current_job.get('error'))}")
        print(f"{'='*60}\n")
        
        with job_lock:
            current_job["running"] = False


def start_analysis(message: str, history: list):
    """Start analysis - returns immediately with 'processing' message."""
    global current_job, state
    
    print(f"\n[START_ANALYSIS] Called with message: {message[:50]}")
    
    if not message.strip():
        return "", history, [], "⚠️ Please enter a question"
    
    # Check if already running
    with job_lock:
        if current_job["running"]:
            elapsed = int(time.time() - current_job["start_time"]) if current_job["start_time"] else 0
            return "", history, [], f"⏳ Analysis already in progress... ({elapsed}s)\n{current_job['progress']}"
    
    print(f"[START_ANALYSIS] Starting new analysis")
    
    # Add user message to history
    history = history + [(message, "⏳ Processing... This may take 15-20 minutes for deep analysis.\n\nClick 'Check Status' to see progress.")]
    
    # Create thread-safe copy of state
    with state_lock:
        thread_state = copy.deepcopy(state)
    
    print(f"[START_ANALYSIS] State copied. Turn count: {thread_state.get('turn_count', 0)}")
    
    # Start background thread
    thread = threading.Thread(
        target=run_analysis_thread, 
        args=(message, thread_state),
        name=f"Analysis-{int(time.time())}"
    )
    thread.daemon = True
    thread.start()
    
    print(f"[START_ANALYSIS] Thread started: {thread.name} (ID: {thread.ident})")
    
    # Give thread a moment to start
    time.sleep(0.5)
    
    return "", history, [], "⏳ Analysis started in background..."


def check_status(history: list):
    """Check job status and update response when complete."""
    global current_job, captured_charts
    
    # #region agent log - Hypothesis C/E: check captured_charts state in check_status
    print(f"[DEBUG-CHECK] check_status called")
    print(f"[DEBUG-CHECK] captured_charts count: {len(captured_charts)}")
    print(f"[DEBUG-CHECK] captured_charts id: {id(captured_charts)}")
    print(f"[DEBUG-CHECK] captured_charts types: {[type(c).__name__ for c in captured_charts]}")
    # #endregion
    
    if not history:
        return history, [], "No analysis running"
    
    with job_lock:
        is_running = current_job["running"]
        progress = current_job["progress"]
        result = current_job["result"]
        error = current_job["error"]
        start_time = current_job["start_time"]
        thread_id = current_job["thread_id"]
    
    elapsed = int(time.time() - start_time) if start_time else 0
    mins, secs = divmod(elapsed, 60)
    
    if is_running:
        # Still running - show progress
        progress_bar = "█" * min(20, elapsed // 30) + "░" * max(0, 20 - elapsed // 30)
        status = f"⏳ Running ({mins}m {secs}s elapsed)\n[{progress_bar}]\n\nThread ID: {thread_id}\nStatus: {progress}"
        print(f"[CHECK_STATUS] Still running... {mins}m {secs}s")
        # #region agent log - Hypothesis E: what we return to Gradio
        print(f"[DEBUG-CHECK] Returning to Gradio (running): charts count = {len(captured_charts)}")
        # #endregion
        return history, captured_charts, status
    
    elif result:
        # Complete - update the last message
        print(f"[CHECK_STATUS] Complete! Updating history")
        if history and history[-1][1] and history[-1][1].startswith("⏳"):
            history[-1] = (history[-1][0], result)
        status = f"✅ Complete! ({mins}m {secs}s total)\n📊 Charts: {len(captured_charts)}\nClick 'Check Status' again to refresh charts."
        # #region agent log - Hypothesis E: what we return to Gradio
        print(f"[DEBUG-CHECK] Returning to Gradio (complete): charts count = {len(captured_charts)}")
        for i, c in enumerate(captured_charts):
            print(f"[DEBUG-CHECK]   Chart {i}: type={type(c).__name__}, size={getattr(c, 'size', 'N/A')}")
        # #endregion
        return history, captured_charts, status
    
    elif error:
        # Error
        print(f"[CHECK_STATUS] Error detected: {error}")
        if history and history[-1][1] and history[-1][1].startswith("⏳"):
            history[-1] = (history[-1][0], f"❌ Error: {error}")
        status = f"❌ Error after {mins}m {secs}s"
        return history, captured_charts, status
    
    else:
        return history, captured_charts, "Ready - enter a question"


def clear_chat():
    """Clear chat and reset state."""
    global state, current_job, captured_charts
    
    print("[CLEAR_CHAT] Resetting...")
    
    with state_lock:
        state = coordinator.get_initial_state()
    
    with job_lock:
        current_job = {
            "running": False,
            "progress": "",
            "result": None,
            "error": None,
            "start_time": None,
            "thread_id": None
        }
    
    captured_charts = []
    
    print("[CLEAR_CHAT] Done")
    return [], [], "Cleared - ready for new questions"


# ==============================================
# GRADIO INTERFACE
# ==============================================

with gr.Blocks(title=f"Hotel Intelligence: {HOTEL_NAME}") as demo:
    gr.Markdown(f"# 🏨 Hotel Intelligence: {HOTEL_NAME}")
    gr.Markdown(f"""
    *Supports long-running ML analysis (15-20 min). Click 'Check Status' to see progress.*
    
    **Quick queries** (< 1 min): Reviews, rankings, competitor info  
    **Deep analysis** (15-20 min): Feature impact, NLP comparison
    
    **Hotel:** {HOTEL_NAME} (ID: {HOTEL_ID})  
    **City:** {CITY}
    """)
    
    chatbot = gr.Chatbot(height=400, label="Conversation")
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question... (e.g., 'What features should I improve?')", 
            show_label=False, 
            scale=4
        )
        send_btn = gr.Button("🚀 Send", variant="primary")
    
    with gr.Row():
        check_btn = gr.Button("🔄 Check Status", variant="secondary")
        clear_btn = gr.Button("🗑️ Clear Chat", variant="stop")
    
    status_box = gr.Textbox(
        label="Status", 
        value="Ready - enter a question",
        interactive=False,
        lines=5
    )
    
    # Chart gallery
    chart_gallery = gr.Gallery(
        label="📊 Analysis Charts", 
        columns=2, 
        height=300, 
        visible=True
    )
    
    # Button handlers
    send_btn.click(
        fn=start_analysis,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, chart_gallery, status_box]
    )
    
    msg.submit(
        fn=start_analysis,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, chart_gallery, status_box]
    )
    
    check_btn.click(
        fn=check_status,
        inputs=[chatbot],
        outputs=[chatbot, chart_gallery, status_box]
    )
    
    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, chart_gallery, status_box]
    )
    
    gr.Markdown("""
    ---
    ### Example Questions
    - "What are guests saying about wifi?" (quick)
    - "How do I compare to competitors?" (quick)
    - "Am I more responsive to my guests compared to my competitors?" (medium)
    - "What features should I improve to increase my rating?" (deep - 15 min)
    
    ### Troubleshooting
    - If stuck: Check the cell output logs for [THREAD] messages
    - Status not updating: Click 'Check Status' button manually
    - Charts not showing: Click 'Check Status' after completion
    
    *Charts appear above when running feature impact analysis*
    """)

print("\n[INIT] Launching Gradio interface...")
demo.queue()
demo.launch(share=True)
print("[INIT] Gradio launched successfully")
