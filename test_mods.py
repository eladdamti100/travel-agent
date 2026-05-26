#!/usr/bin/env python
"""
Quick test script to validate modification detector and cache checker changes
"""
import sys
from src.utils.modification_detector import detect_modification_context, extract_modified_parameters

print("=" * 60)
print("Testing Modification Detector")
print("=" * 60)

# Test cases for modification detection
test_cases = [
    ("change the origin airport to TLV", True, "airport change"),
    ("plan me a trip to london", False, "new trip plan"),
    ("what hotels are available", False, "research query"),
    ("different airport", True, "airport modification"),
    ("TLV", False, "just airport code (HITL answer)"),
    ("update budget to $3000", True, "budget modification"),
    ("7 days", False, "just duration (HITL answer)"),
    ("from JFK to TLV", True, "route change"),
    ("make it 10 days instead", True, "duration modification"),
]

print("\nModification Detection Tests:")
print("-" * 60)

for msg, expected, desc in test_cases:
    result = detect_modification_context(msg)
    status = "✓" if result == expected else "✗"
    print(f"{status} {desc:30} | '{msg}' -> {result}")
    if result != expected:
        print(f"  Expected: {expected}, Got: {result}")
        sys.exit(1)

print("\n" + "=" * 60)
print("Testing Parameter Extraction")
print("=" * 60)

extract_test_cases = [
    ("change the origin airport to TLV", "origin_airport", "TLV"),
    ("update budget to $3000", "budget", 3000.0),
    ("modify duration to 10 days", "duration", "10 days"),
]

print("\nParameter Extraction Tests:")
print("-" * 60)

for msg, param_key, expected_val in extract_test_cases:
    params = extract_modified_parameters(msg)
    actual_val = params.get(param_key)
    status = "✓" if actual_val == expected_val else "✗"
    print(f"{status} {param_key:20} | '{msg}'")
    print(f"  -> {actual_val}")
    if actual_val != expected_val:
        print(f"  Expected: {expected_val}")
        sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
