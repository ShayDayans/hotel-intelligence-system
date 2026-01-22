"""
Tool Call Recovery Test for Databricks

Tests the malformed tool call recovery functionality that handles
Groq API "tool_use_failed" errors by extracting and parsing the
failed_generation content.

Run this on Databricks to verify the fix works in production.
"""

import os
import sys
from datetime import datetime

# ============================================
# 1. ENVIRONMENT CHECK
# ============================================
print("=" * 60)
print("TOOL CALL RECOVERY TEST")
print("=" * 60)
print(f"Test started at: {datetime.now()}")
print()

on_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ
print(f"Running on Databricks: {on_databricks}")
if on_databricks:
    print(f"Databricks Runtime: {os.environ.get('DATABRICKS_RUNTIME_VERSION', 'N/A')}")
print()

# ============================================
# 2. IMPORT TEST
# ============================================
print("-" * 60)
print("Testing imports...")
print("-" * 60)

try:
    from agents.base_agent import BaseAgent
    print("[OK] Successfully imported BaseAgent")
except ImportError as e:
    print(f"[ERROR] Failed to import BaseAgent: {e}")
    sys.exit(1)

try:
    from agents.review_analyst import ReviewAnalystAgent
    print("[OK] Successfully imported ReviewAnalystAgent")
except ImportError as e:
    print(f"[ERROR] Failed to import ReviewAnalystAgent: {e}")
    sys.exit(1)

try:
    from agents.coordinator import LangGraphCoordinator
    print("[OK] Successfully imported LangGraphCoordinator")
except ImportError as e:
    print(f"[ERROR] Failed to import LangGraphCoordinator: {e}")
    sys.exit(1)

print()

# ============================================
# 3. TEST CONFIGURATION
# ============================================
print("-" * 60)
print("Test Configuration")
print("-" * 60)

# Use Airbnb property for full testing
TEST_HOTEL_ID = "ABB_40458495"  # Update if needed
TEST_HOTEL_NAME = "Test Airbnb Property"
TEST_CITY = "London"

print(f"Hotel ID: {TEST_HOTEL_ID}")
print(f"Hotel Name: {TEST_HOTEL_NAME}")
print(f"City: {TEST_CITY}")
print()

# ============================================
# 4. TEST: _parse_malformed_tool_calls
# ============================================
print("=" * 60)
print("TEST 1: Malformed Tool Call Parser")
print("=" * 60)

# Create a minimal agent to test the parser
class TestAgent(BaseAgent):
    def get_system_prompt(self):
        return "Test agent"
    def get_tools(self):
        return []

agent = TestAgent(
    hotel_id=TEST_HOTEL_ID,
    hotel_name=TEST_HOTEL_NAME,
    city=TEST_CITY
)

# Test cases matching the actual error format from Groq
test_cases = [
    # Format seen in the error: <function=name {"arg": "val"}'
    {
        "name": "Groq API error format (no closing tag)",
        "content": '<function=search_airbnb_reviews {"query": "wifi", "k": 10}\'',
        "expected_tool": "search_airbnb_reviews",
        "expected_args": {"query": "wifi", "k": 10}
    },
    # Standard format with closing tag
    {
        "name": "Standard format with closing tag",
        "content": '<function=search_booking_reviews {"query": "breakfast", "k": 5}</function>',
        "expected_tool": "search_booking_reviews",
        "expected_args": {"query": "breakfast", "k": 5}
    },
    # Format with space before args
    {
        "name": "Format with space before JSON",
        "content": '<function=analyze_sentiment_topics {"topics": ["wifi", "noise"]}',
        "expected_tool": "analyze_sentiment_topics",
        "expected_args": {"topics": ["wifi", "noise"]}
    },
]

# Fake tool map for testing
fake_tool_map = {
    "search_airbnb_reviews": lambda **x: x,
    "search_booking_reviews": lambda **x: x,
    "analyze_sentiment_topics": lambda **x: x,
}

all_passed = True

for tc in test_cases:
    print(f"\n--- Testing: {tc['name']} ---")
    print(f"Input: {tc['content'][:60]}...")
    
    result = agent._parse_malformed_tool_calls(tc["content"], fake_tool_map)
    
    if result:
        parsed = result[0]
        name_match = parsed["name"] == tc["expected_tool"]
        
        # Check args (may have type coercion)
        args_match = True
        for key, expected_val in tc["expected_args"].items():
            if key not in parsed["args"]:
                args_match = False
                break
            actual_val = parsed["args"][key]
            # Handle type coercion (strings to ints)
            if isinstance(expected_val, int) and isinstance(actual_val, str):
                args_match = actual_val == str(expected_val)
            elif actual_val != expected_val:
                args_match = False
        
        if name_match and args_match:
            print(f"[OK] Parsed correctly: {parsed['name']}({parsed['args']})")
        else:
            print(f"[WARN] Mismatch:")
            print(f"  Expected: {tc['expected_tool']}({tc['expected_args']})")
            print(f"  Got: {parsed['name']}({parsed['args']})")
            all_passed = False
    else:
        print(f"[FAIL] Could not parse")
        all_passed = False

print()
if all_passed:
    print("[OK] TEST 1 PASSED: All malformed formats parsed correctly")
else:
    print("[WARN] TEST 1 PARTIAL: Some formats failed to parse")

# ============================================
# 5. TEST: Full Agent Query (Review Analyst)
# ============================================
print()
print("=" * 60)
print("TEST 2: Full Agent Query (Review Analyst)")
print("=" * 60)

try:
    review_agent = ReviewAnalystAgent(
        hotel_id=TEST_HOTEL_ID,
        hotel_name=TEST_HOTEL_NAME,
        city=TEST_CITY
    )
    
    print("[OK] ReviewAnalystAgent initialized")
    
    # Query that will trigger tool use
    test_query = "What are guests saying about wifi quality?"
    print(f"\nQuery: {test_query}")
    print("Processing... (this may take a moment)")
    print()
    
    response = review_agent.run(test_query)
    
    print("-" * 60)
    print("Response (first 500 chars):")
    print("-" * 60)
    print(response[:500])
    print()
    
    # Check for success indicators
    has_content = len(response) > 50
    no_error = "error" not in response.lower()[:100]
    no_failed = "failed" not in response.lower()[:100]
    
    if has_content and no_error and no_failed:
        print("[OK] TEST 2 PASSED: Agent returned valid response")
    else:
        print("[WARN] TEST 2 UNCERTAIN: Response may indicate issues")
        if not has_content:
            print("  - Response too short")
        if not no_error:
            print("  - Response contains 'error'")
        if not no_failed:
            print("  - Response contains 'failed'")
        
except Exception as e:
    print(f"[ERROR] TEST 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# 6. TEST: Coordinator Full Pipeline
# ============================================
print()
print("=" * 60)
print("TEST 3: Coordinator Full Pipeline")
print("=" * 60)

try:
    coordinator = LangGraphCoordinator(
        hotel_id=TEST_HOTEL_ID,
        hotel_name=TEST_HOTEL_NAME,
        city=TEST_CITY
    )
    
    print("[OK] Coordinator initialized")
    
    # Query that will trigger Review Analyst
    test_query = "What reviews mention wifi or internet?"
    print(f"\nQuery: {test_query}")
    print("Processing through full pipeline...")
    print()
    
    # LangGraphCoordinator.run() returns (response, state) tuple
    response, state = coordinator.run(test_query)
    
    print("-" * 60)
    print("Response (first 800 chars):")
    print("-" * 60)
    print(response[:800] if response else "No response")
    print()
    
    # Check for success
    has_content = response and len(response) > 100
    no_error = not response or "agent execution failed" not in response.lower()
    no_tool_failed = not response or "tool_use_failed" not in response.lower()
    
    if has_content and no_error and no_tool_failed:
        print("[OK] TEST 3 PASSED: Coordinator returned valid response")
    else:
        print("[WARN] TEST 3 UNCERTAIN: Response may indicate issues")
        
except Exception as e:
    print(f"[ERROR] TEST 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# 7. SUMMARY
# ============================================
print()
print("=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Test completed at: {datetime.now()}")
print()
print("What was tested:")
print("  1. Malformed tool call parser (_parse_malformed_tool_calls)")
print("  2. Full agent query with ReviewAnalystAgent")
print("  3. Full pipeline with LangGraphCoordinator")
print()
print("The fix handles:")
print("  - Groq API 'tool_use_failed' errors")
print("  - Malformed tool calls like: <function=name {args}'>")
print("  - Recovery by parsing failed_generation content")
print("  - Continuing conversation with HumanMessage for results")
print()
print("=" * 60)
