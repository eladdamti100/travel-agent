#!/usr/bin/env python
"""
Integration test to verify the modification detection flow works end-to-end
"""
import sys
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage

# Test 1: Import all modules to check for syntax errors
print("=" * 60)
print("Test 1: Checking all modules can be imported")
print("=" * 60)

try:
    from src.utils.modification_detector import detect_modification_context
    print("✓ Imported modification_detector")
    
    from src.graph.state import AgentState
    print("✓ Imported AgentState")
    
    from src.agents.cache_checker import run_cache_check
    print("✓ Imported cache_checker")
    
    from src.graph.nodes import extract_metadata
    print("✓ Imported nodes.extract_metadata")
    
    from src.models.cache import CacheStatus
    print("✓ Imported CacheStatus")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test extract_metadata with modification detection
print("\n" + "=" * 60)
print("Test 2: Testing extract_metadata modification detection")
print("=" * 60)

state = {
    "messages": [HumanMessage(content="change the origin airport to TLV")],
    "current_city": None,
    "total_budget": None,
}

result = extract_metadata(state)
print(f"Input: 'change the origin airport to TLV'")
print(f"Result: {result}")

if result.get("force_replan") is True:
    print("✓ force_replan correctly set to True")
else:
    print(f"✗ force_replan should be True, got {result.get('force_replan')}")
    sys.exit(1)

# Test 3: Test cache_checker respects force_replan
print("\n" + "=" * 60)
print("Test 3: Testing cache_checker respects force_replan")
print("=" * 60)

# Mock find_cached_answer to return a cache hit
hit_result = MagicMock()
hit_result.status = CacheStatus.HIT
hit_result.similarity_score = 0.95
hit_result.matched_query = "Plan a trip to London"
hit_result.cached_answer = "Here is your London trip"

state_with_force_replan = {
    "messages": [HumanMessage(content="change airport to TLV")],
    "force_replan": True,
}

with patch("src.agents.cache_checker.find_cached_answer", return_value=hit_result):
    result = run_cache_check(state_with_force_replan)

print(f"State: force_replan=True, would otherwise get cache HIT")
print(f"Result cache_status: {result.get('cache_status')}")

if result.get("cache_status") == CacheStatus.MISS.value:
    print("✓ Cache hit correctly bypassed when force_replan=True")
else:
    print(f"✗ Expected cache_status='miss', got {result.get('cache_status')}")
    sys.exit(1)

# Test 4: Test normal cache behavior without force_replan
print("\n" + "=" * 60)
print("Test 4: Testing normal cache behavior (no force_replan)")
print("=" * 60)

state_normal = {
    "messages": [HumanMessage(content="plan a trip to london")],
    "force_replan": False,
}

with patch("src.agents.cache_checker.find_cached_answer", return_value=hit_result):
    result = run_cache_check(state_normal)

print(f"State: force_replan=False, normal cache operation")
print(f"Result cache_status: {result.get('cache_status')}")

if result.get("cache_status") == CacheStatus.HIT.value:
    print("✓ Cache hit works normally when force_replan=False")
else:
    print(f"✗ Expected cache_status='hit', got {result.get('cache_status')}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All integration tests passed! ✓")
print("=" * 60)
