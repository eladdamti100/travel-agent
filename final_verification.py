#!/usr/bin/env python
"""
Final verification that all changes work together correctly.
This tests the complete flow for modification detection and re-planning.
"""
import sys
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage

print("=" * 70)
print("FINAL VERIFICATION: Modification Detection & Cache Bypass")
print("=" * 70)

# Test 1: Verify all imports
print("\n[1/5] Verifying imports...")
try:
    from src.utils.modification_detector import detect_modification_context
    from src.graph.state import AgentState
    from src.agents.cache_checker import run_cache_check
    from src.graph.nodes import extract_metadata
    from src.models.cache import CacheStatus
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test modification detection
print("\n[2/5] Testing modification detection...")
test_msgs = [
    ("change the origin airport to TLV", True),
    ("plan me a trip to london", False),
    ("update budget to $3000", True),
    ("TLV", False),
]

for msg, should_detect in test_msgs:
    result = detect_modification_context(msg)
    if result == should_detect:
        print(f"✓ '{msg[:40]}...' -> {result}")
    else:
        print(f"✗ '{msg[:40]}...' expected {should_detect}, got {result}")
        sys.exit(1)

# Test 3: Test extract_metadata with modification
print("\n[3/5] Testing extract_metadata modification detection...")
state = {"messages": [HumanMessage(content="change the origin airport to TLV")]}
result = extract_metadata(state)
if result.get("force_replan") is True:
    print(f"✓ extract_metadata set force_replan=True for modification")
else:
    print(f"✗ extract_metadata should set force_replan=True")
    sys.exit(1)

# Test 4: Test extract_metadata without modification
print("\n[4/5] Testing extract_metadata without modification...")
state = {"messages": [HumanMessage(content="plan me a trip to london")]}
result = extract_metadata(state)
if result.get("force_replan") is False:
    print(f"✓ extract_metadata set force_replan=False for new trip")
else:
    print(f"✗ extract_metadata should set force_replan=False")
    sys.exit(1)

# Test 5: Test cache_checker respects force_replan
print("\n[5/5] Testing cache_checker respects force_replan flag...")

# Mock a cache hit
cache_hit_result = MagicMock()
cache_hit_result.status = CacheStatus.HIT
cache_hit_result.similarity_score = 0.95
cache_hit_result.matched_query = "Plan a trip to London"
cache_hit_result.cached_answer = "Here is your London plan"

# Test with force_replan=True (should bypass cache)
state = {
    "messages": [HumanMessage(content="change airport to TLV")],
    "force_replan": True,
}

with patch("src.agents.cache_checker.find_cached_answer", return_value=cache_hit_result):
    result = run_cache_check(state)
    if result["cache_status"] == CacheStatus.MISS.value:
        print("✓ Cache hit bypassed when force_replan=True")
    else:
        print(f"✗ Expected cache MISS when force_replan=True, got {result['cache_status']}")
        sys.exit(1)

# Test with force_replan=False (should allow cache hit)
state = {
    "messages": [HumanMessage(content="plan me a trip to london")],
    "force_replan": False,
}

with patch("src.agents.cache_checker.find_cached_answer", return_value=cache_hit_result):
    result = run_cache_check(state)
    if result["cache_status"] == CacheStatus.HIT.value:
        print("✓ Cache hit allowed when force_replan=False")
    else:
        print(f"✗ Expected cache HIT when force_replan=False, got {result['cache_status']}")
        sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL VERIFICATIONS PASSED!")
print("=" * 70)
print("\nThe fix is working correctly:")
print("1. Modification detection identifies when users modify trip parameters")
print("2. extract_metadata sets force_replan flag appropriately")
print("3. cache_checker bypasses cache when force_replan=True")
print("4. Normal cache behavior preserved when force_replan=False")
print("\nWhen user says 'change the origin airport to TLV':")
print("→ System detects it's a modification")
print("→ Sets force_replan=True")
print("→ cache_checker bypasses cache")
print("→ master_planner creates new trip with TLV as origin")
print("→ User gets re-planned trip with new airport ✓")
