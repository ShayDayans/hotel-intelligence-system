"""
Test All Agents - Comprehensive System Test

Tests all components of the Hotel Intelligence System:
- Agent initialization and tools
- Coordinator routing
- Market Intel (weather, events via BrightData)
- Review Analyst (RAG search)
- Chart URL storage
- Multi-agent state management

Note: Skips NLP and LR Databricks tools (tested separately on cluster)
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()


def test_decorator(func):
    """Decorator to run tests with error handling."""
    def wrapper():
        print(f"\n{'='*60}")
        print(f"TEST: {func.__name__.replace('_', ' ').title()}")
        print('='*60)
        try:
            func()
            print(f"✅ PASS: {func.__name__}")
            return True
        except Exception as e:
            print(f"❌ FAIL: {func.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    return wrapper


# ===========================================
# AGENT INITIALIZATION TESTS
# ===========================================

@test_decorator
def test_review_analyst_init():
    """Test ReviewAnalystAgent initialization."""
    from review_analyst import ReviewAnalystAgent
    
    agent = ReviewAnalystAgent(
        hotel_id="ABB_40458495",
        hotel_name="Rental unit in Broadbeach",
        city="Broadbeach"
    )
    
    tools = agent.get_tools()
    tool_names = [t.name for t in tools]
    
    print(f"   Tools: {tool_names}")
    assert len(tools) >= 3, f"Expected at least 3 tools, got {len(tools)}"
    assert any("review" in t.lower() for t in tool_names), "Missing review search tool"
    
    print(f"   ✓ Agent initialized with {len(tools)} tools")


@test_decorator
def test_competitor_analyst_init():
    """Test CompetitorAnalystAgent initialization."""
    from competitor_analyst import CompetitorAnalystAgent
    
    agent = CompetitorAnalystAgent(
        hotel_id="ABB_40458495",
        hotel_name="Rental unit in Broadbeach",
        city="Broadbeach"
    )
    
    tools = agent.get_tools()
    tool_names = [t.name for t in tools]
    
    print(f"   Tools: {tool_names}")
    assert len(tools) >= 2, f"Expected at least 2 tools, got {len(tools)}"
    
    print(f"   ✓ Agent initialized with {len(tools)} tools")


@test_decorator
def test_market_intel_init():
    """Test MarketIntelAgent initialization."""
    from market_intel import MarketIntelAgent
    
    agent = MarketIntelAgent(
        hotel_id="ABB_40458495",
        hotel_name="Rental unit in Broadbeach",
        city="Broadbeach"
    )
    
    tools = agent.get_tools()
    tool_names = [t.name for t in tools]
    
    print(f"   Tools: {tool_names}")
    assert len(tools) >= 2, f"Expected at least 2 tools, got {len(tools)}"
    assert any("weather" in t.lower() for t in tool_names), "Missing weather tool"
    assert any("event" in t.lower() for t in tool_names), "Missing event tool"
    
    print(f"   ✓ Agent initialized with {len(tools)} tools")


@test_decorator
def test_benchmark_agent_init():
    """Test BenchmarkAgent initialization."""
    from benchmark_agent import BenchmarkAgent
    
    agent = BenchmarkAgent(
        hotel_id="ABB_40458495",
        hotel_name="Rental unit in Broadbeach",
        city="Broadbeach"
    )
    
    tools = agent.get_tools()
    tool_names = [t.name for t in tools]
    
    print(f"   Tools: {tool_names}")
    assert len(tools) >= 2, f"Expected at least 2 tools, got {len(tools)}"
    
    print(f"   ✓ Agent initialized with {len(tools)} tools")


# ===========================================
# COORDINATOR TESTS
# ===========================================

@test_decorator
def test_coordinator_init():
    """Test LangGraphCoordinator initialization."""
    from coordinator import LangGraphCoordinator
    
    coordinator = LangGraphCoordinator(
        hotel_id="ABB_40458495",
        hotel_name="Rental unit in Broadbeach",
        city="Broadbeach"
    )
    
    # Check all agents are initialized
    assert "review_analyst" in coordinator.agents
    assert "competitor_analyst" in coordinator.agents
    assert "market_intel" in coordinator.agents
    assert "benchmark_agent" in coordinator.agents
    
    print(f"   ✓ All 4 specialist agents initialized")
    
    # Check graph is built
    assert coordinator.graph is not None
    print(f"   ✓ LangGraph workflow compiled")


@test_decorator
def test_coordinator_initial_state():
    """Test coordinator initial state structure."""
    from coordinator import LangGraphCoordinator
    
    coordinator = LangGraphCoordinator(
        hotel_id="ABB_40458495",
        hotel_name="Rental unit in Broadbeach",
        city="Broadbeach"
    )
    
    state = coordinator.get_initial_state()
    
    # Check required fields
    required_fields = [
        "query", "selected_agent", "agent_queue", "intermediate_results",
        "agents_executed", "response", "recent_turns", "summary",
        "entities", "hotel_id", "hotel_name", "city", "turn_count"
    ]
    
    for field in required_fields:
        assert field in state, f"Missing field: {field}"
    
    print(f"   ✓ All {len(required_fields)} required fields present")
    
    # Check initial values
    assert state["agent_queue"] == []
    assert state["agents_executed"] == []
    assert state["hotel_id"] == "ABB_40458495"
    
    print(f"   ✓ Initial values correct")


@test_decorator
def test_coordinator_routing_prompt():
    """Test routing prompt contains expected patterns."""
    from coordinator import LangGraphCoordinator
    
    coordinator = LangGraphCoordinator(
        hotel_id="ABB_40458495",
        hotel_name="Rental unit in Broadbeach",
        city="Broadbeach"
    )
    
    prompt = coordinator.ROUTING_PROMPT
    
    # Check agent mentions
    assert "review_analyst" in prompt
    assert "competitor_analyst" in prompt
    assert "benchmark_agent" in prompt
    assert "market_intel" in prompt
    
    print(f"   ✓ All agents mentioned in routing prompt")
    
    # Check routing patterns
    assert "weather" in prompt.lower()
    assert "event" in prompt.lower()
    
    print(f"   ✓ Routing patterns present")


@test_decorator
def test_should_continue_logic():
    """Test the _should_continue conditional logic."""
    from coordinator import LangGraphCoordinator
    
    coordinator = LangGraphCoordinator(
        hotel_id="ABB_40458495",
        hotel_name="Rental unit in Broadbeach",
        city="Broadbeach"
    )
    
    # Test: No selected agent -> aggregate
    state_empty = {"selected_agent": "", "agents_executed": []}
    result = coordinator._should_continue(state_empty)
    assert result == "aggregate", f"Expected 'aggregate', got '{result}'"
    print(f"   ✓ No selected agent -> 'aggregate'")
    
    # Test: Agent already executed -> aggregate
    state_executed = {"selected_agent": "review_analyst", "agents_executed": ["review_analyst"]}
    result = coordinator._should_continue(state_executed)
    assert result == "aggregate", f"Expected 'aggregate', got '{result}'"
    print(f"   ✓ Agent executed -> 'aggregate'")
    
    # Test: Agent pending -> continue
    state_pending = {"selected_agent": "benchmark_agent", "agents_executed": ["review_analyst"]}
    result = coordinator._should_continue(state_pending)
    assert result == "continue", f"Expected 'continue', got '{result}'"
    print(f"   ✓ Agent pending -> 'continue'")


# ===========================================
# DATABRICKS TOOLS TESTS (non-Databricks parts)
# ===========================================

@test_decorator
def test_databricks_tools_helpers():
    """Test databricks_tools helper functions."""
    from databricks_tools import extract_raw_id, get_source, is_airbnb_property
    
    # Test extract_raw_id
    assert extract_raw_id("ABB_40458495") == "40458495"
    assert extract_raw_id("BKG_177691") == "177691"
    print(f"   ✓ extract_raw_id works")
    
    # Test get_source
    assert get_source("ABB_40458495") == "airbnb"
    assert get_source("BKG_177691") == "booking"
    assert get_source("UNKNOWN_123") == "unknown"
    print(f"   ✓ get_source works")
    
    # Test is_airbnb_property
    assert is_airbnb_property("ABB_40458495") == True
    assert is_airbnb_property("BKG_177691") == False
    print(f"   ✓ is_airbnb_property works")


@test_decorator
def test_chart_url_storage():
    """Test chart URL storage mechanism."""
    from databricks_tools import get_chart_urls, clear_chart_urls, _last_chart_urls
    
    # Clear and check empty
    clear_chart_urls()
    urls = get_chart_urls()
    assert urls == [], f"Expected empty list, got {urls}"
    print(f"   ✓ clear_chart_urls works")
    
    # get_chart_urls returns a copy
    urls1 = get_chart_urls()
    urls2 = get_chart_urls()
    assert urls1 is not urls2, "get_chart_urls should return a copy"
    print(f"   ✓ get_chart_urls returns copy")


# ===========================================
# ENTITY EXTRACTION TESTS
# ===========================================

@test_decorator
def test_entity_extractor():
    """Test entity extraction from queries."""
    from entity_extractor import extract_entities
    
    # Test competitor detection
    entities = extract_entities("How do I compare to my competitors?", use_llm=False)
    assert "competitors" in entities.to_dict()
    print(f"   ✓ Competitor keywords detected")
    
    # Test metric detection
    entities = extract_entities("What is my rating compared to others?", use_llm=False)
    assert "metrics" in entities.to_dict()
    print(f"   ✓ Metric keywords detected")
    
    # Test topic detection
    entities = extract_entities("What do guests say about wifi?", use_llm=False)
    assert "topics" in entities.to_dict()
    print(f"   ✓ Topic keywords detected")


# ===========================================
# MEMORY MANAGER TESTS
# ===========================================

@test_decorator
def test_memory_manager():
    """Test memory manager functions."""
    from memory_manager import get_context_for_agent
    from graph_state import AgentState, ConversationTurn
    
    # Create state with some history
    state = AgentState(
        query="What is my rating?",
        selected_agent="benchmark_agent",
        agent_queue=[],
        intermediate_results=[],
        agents_executed=[],
        response="",
        recent_turns=[
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hi there!"),
        ],
        summary="User asked about hotel.",
        entities={},
        hotel_id="ABB_40458495",
        hotel_name="Test Hotel",
        city="Test City",
        turn_count=2
    )
    
    context = get_context_for_agent(state)
    
    assert "Hello" in context or "Hi there" in context or len(context) > 0
    print(f"   ✓ Context generated from state")


# ===========================================
# GRAPH STATE TESTS
# ===========================================

@test_decorator
def test_graph_state_structure():
    """Test AgentState TypedDict structure."""
    from graph_state import AgentState, ConversationTurn, ExtractedEntities
    
    # Check AgentState fields
    annotations = AgentState.__annotations__
    
    required = ["query", "selected_agent", "agent_queue", "response", "hotel_id"]
    for field in required:
        assert field in annotations, f"Missing field: {field}"
    
    print(f"   ✓ AgentState has {len(annotations)} fields")
    
    # Check ConversationTurn
    turn = ConversationTurn(role="user", content="test")
    assert turn.role == "user"
    assert turn.content == "test"
    print(f"   ✓ ConversationTurn works")
    
    # Check ExtractedEntities
    entities = ExtractedEntities()
    assert hasattr(entities, "to_dict")
    print(f"   ✓ ExtractedEntities works")


# ===========================================
# LLM FALLBACK TESTS
# ===========================================

@test_decorator
def test_llm_fallback_init():
    """Test LLM fallback initialization."""
    from base_agent import LLMWithFallback
    
    llm = LLMWithFallback()
    
    # Check that LLM is initialized (either primary or fallback)
    assert llm.llm is not None, "LLM should be initialized"
    print(f"   ✓ LLM initialized successfully")


# ===========================================
# REVIEW ANALYST TOOL TESTS
# ===========================================

@test_decorator
def test_review_analyst_rag_search():
    """Test ReviewAnalystAgent RAG search tool."""
    from review_analyst import ReviewAnalystAgent
    
    agent = ReviewAnalystAgent(
        hotel_id="ABB_40458495",
        hotel_name="Rental unit in Broadbeach",
        city="Broadbeach"
    )
    
    # Test search (may return no results if DB empty, but shouldn't error)
    result = agent.search_airbnb_reviews(topic="location", k=3)
    
    assert isinstance(result, str), "Result should be a string"
    print(f"   ✓ search_airbnb_reviews returned: {result[:80]}...")


@test_decorator
def test_review_analyst_booking_search():
    """Test ReviewAnalystAgent Booking search tool."""
    from review_analyst import ReviewAnalystAgent
    
    agent = ReviewAnalystAgent(
        hotel_id="BKG_177691",
        hotel_name="Malmaison London",
        city="London"
    )
    
    # Test search
    result = agent.search_booking_reviews(topic="breakfast", k=3)
    
    assert isinstance(result, str), "Result should be a string"
    print(f"   ✓ search_booking_reviews returned: {result[:80]}...")


# ===========================================
# BENCHMARK AGENT TOOL TESTS
# ===========================================

@test_decorator
def test_benchmark_agent_rank_tool():
    """Test BenchmarkAgent rank_by_metric tool."""
    from benchmark_agent import BenchmarkAgent
    
    agent = BenchmarkAgent(
        hotel_id="ABB_40458495",
        hotel_name="Rental unit in Broadbeach",
        city="Broadbeach"
    )
    
    result = agent.rank_by_metric(metric="rating", k=5)
    
    assert isinstance(result, str), "Result should be a string"
    print(f"   ✓ rank_by_metric returned: {result[:80]}...")


# ===========================================
# COMPETITOR ANALYST TOOL TESTS
# ===========================================

@test_decorator
def test_competitor_analyst_geo_search():
    """Test CompetitorAnalystAgent geo search tool."""
    from competitor_analyst import CompetitorAnalystAgent
    
    agent = CompetitorAnalystAgent(
        hotel_id="ABB_40458495",
        hotel_name="Rental unit in Broadbeach",
        city="Broadbeach"
    )
    
    result = agent.find_competitors_geo(city="Broadbeach", k=3)
    
    assert isinstance(result, str), "Result should be a string"
    print(f"   ✓ find_competitors_geo returned: {result[:80]}...")


# ===========================================
# MARKET INTEL TOOL TESTS (Quick - no API calls in basic test)
# ===========================================

@test_decorator
def test_market_intel_has_weather_tool():
    """Test MarketIntelAgent has weather functionality."""
    from market_intel import MarketIntelAgent
    
    agent = MarketIntelAgent(
        hotel_id="ABB_40458495",
        hotel_name="Rental unit in Broadbeach",
        city="Broadbeach"
    )
    
    # Check weather method exists
    assert hasattr(agent, 'get_weather'), "Missing get_weather method"
    print(f"   ✓ get_weather method exists")
    
    # Check event method exists
    assert hasattr(agent, 'search_events'), "Missing search_events method"
    print(f"   ✓ search_events method exists")


# ===========================================
# RUN ALL TESTS
# ===========================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("HOTEL INTELLIGENCE SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*60)
    print("Note: Skipping NLP/LR Databricks tools (test on cluster)")
    
    tests = [
        # Agent initialization
        test_review_analyst_init,
        test_competitor_analyst_init,
        test_market_intel_init,
        test_benchmark_agent_init,
        
        # Coordinator
        test_coordinator_init,
        test_coordinator_initial_state,
        test_coordinator_routing_prompt,
        test_should_continue_logic,
        
        # Databricks tools helpers
        test_databricks_tools_helpers,
        test_chart_url_storage,
        
        # Entity extraction
        test_entity_extractor,
        
        # Memory manager
        test_memory_manager,
        
        # Graph state
        test_graph_state_structure,
        
        # LLM
        test_llm_fallback_init,
        
        # Agent tools
        test_review_analyst_rag_search,
        test_review_analyst_booking_search,
        test_benchmark_agent_rank_tool,
        test_competitor_analyst_geo_search,
        test_market_intel_has_weather_tool,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    passed = sum(results)
    failed = len(results) - passed
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed:  {passed}")
    print(f"❌ Failed:  {failed}")
    print(f"📊 Total:   {len(results)}")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
