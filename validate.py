#!/usr/bin/env python
"""Quick validation of all changes"""
import sys

print("Validating changes...")

try:
    # Test 1: Check imports work
    print("\n1. Testing imports...")
    from src.utils.modification_detector import detect_modification_context, extract_modified_parameters
    from src.graph.state import AgentState
    from src.agents.cache_checker import run_cache_check
    from src.graph.nodes import extract_metadata
    print("   ✓ All imports successful")
    
    # Test 2: Test modification detector on key examples
    print("\n2. Testing modification detection...")
    test_cases = {
        "change the origin airport to TLV": True,
        "plan me a trip to london": False,
        "update budget to $3000": True,
        "7 days": False,
    }
    
    for msg, expected in test_cases.items():
        result = detect_modification_context(msg)
        if result == expected:
            print(f"   ✓ '{msg}' -> {result}")
        else:
            print(f"   ✗ '{msg}' expected {expected} got {result}")
            sys.exit(1)
    
    # Test 3: Test parameter extraction
    print("\n3. Testing parameter extraction...")
    params = extract_modified_parameters("change the origin airport to TLV")
    if params.get("origin_airport") == "TLV":
        print(f"   ✓ Extracted airport code: {params['origin_airport']}")
    else:
        print(f"   ✗ Failed to extract airport code")
        sys.exit(1)
    
    print("\n✓ All validations passed!")
    
except Exception as e:
    print(f"\n✗ Validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
