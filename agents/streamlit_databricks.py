"""
Streamlit Interface for Databricks - Better than Gradio for long jobs

Installation:
%pip install streamlit

Run in Databricks:
%sh streamlit run /Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system/agents/streamlit_databricks.py --server.port 8501
"""

import sys
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system")
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system/agents")

import streamlit as st
import time
from agents.coordinator import LangGraphCoordinator

# ==============================================
# CONFIGURATION
# ==============================================

HOTEL_ID = "ABB_40458495"
HOTEL_NAME = "Rental unit in Broadbeach"
CITY = "Broadbeach"

# ==============================================
# PAGE CONFIG
# ==============================================

st.set_page_config(
    page_title=f"Hotel Intelligence: {HOTEL_NAME}",
    page_icon="🏨",
    layout="wide"
)

# ==============================================
# INITIALIZE SESSION STATE
# ==============================================

if "coordinator" not in st.session_state:
    st.session_state.coordinator = LangGraphCoordinator(HOTEL_ID, HOTEL_NAME, CITY)
    st.session_state.state = st.session_state.coordinator.get_initial_state()
    st.session_state.messages = []
    st.session_state.processing = False

# ==============================================
# HEADER
# ==============================================

st.title(f"🏨 Hotel Intelligence: {HOTEL_NAME}")
st.markdown(f"""
**Hotel:** {HOTEL_NAME} (ID: `{HOTEL_ID}`)  
**City:** {CITY}

*This system supports long-running ML analysis (15-20 min)*
""")

# ==============================================
# SIDEBAR
# ==============================================

with st.sidebar:
    st.header("📊 Query Types")
    st.markdown("""
    **Quick** (< 1 min):
    - Review analysis
    - Rankings
    - Competitor info
    
    **Medium** (2-5 min):
    - Responsiveness comparison
    - Sentiment analysis
    
    **Deep** (15-20 min):
    - Feature impact analysis
    - ML-powered insights
    """)
    
    st.divider()
    
    st.header("💡 Example Questions")
    
    example_queries = [
        "What are guests saying about wifi?",
        "How do I compare to competitors?",
        "Am I more responsive than my competitors?",
        "What features should I improve?"
    ]
    
    for query in example_queries:
        if st.button(query, key=f"example_{query[:20]}"):
            st.session_state.current_query = query
            st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.session_state.state = st.session_state.coordinator.get_initial_state()
        st.rerun()

# ==============================================
# CHAT INTERFACE
# ==============================================

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your hotel..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        # Show processing indicator
        message_placeholder.markdown("⏳ Processing your query...")
        
        # Create progress bar
        progress_bar = progress_placeholder.progress(0)
        status_text = st.empty()
        
        try:
            start_time = time.time()
            
            # Progress updates (visual feedback during long jobs)
            status_text.text("Phase 1: Extracting entities...")
            progress_bar.progress(10)
            time.sleep(0.3)
            
            status_text.text("Phase 2: Routing to specialist agent(s)...")
            progress_bar.progress(20)
            time.sleep(0.3)
            
            status_text.text("Phase 3: Agent analysis (this may take several minutes)...")
            progress_bar.progress(30)
            
            # Run the coordinator (long-running)
            response, st.session_state.state = st.session_state.coordinator.run(
                prompt, 
                st.session_state.state
            )
            
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            
            # Complete
            progress_bar.progress(100)
            status_text.text(f"✅ Complete! ({mins}m {secs}s)")
            time.sleep(1)
            
            # Clear progress indicators
            progress_placeholder.empty()
            status_text.empty()
            
            # Show response
            message_placeholder.markdown(response)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            import traceback
            error_msg = f"❌ **Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            progress_placeholder.empty()
            status_text.empty()

# Handle example query from sidebar
if "current_query" in st.session_state and st.session_state.current_query:
    prompt = st.session_state.current_query
    st.session_state.current_query = None
    st.rerun()
