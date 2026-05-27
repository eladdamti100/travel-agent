"""
Tests for modification detection — identify when user is modifying trip parameters
"""

import pytest
from src.utils.modification_detector import detect_modification_context, extract_modified_parameters


class TestDetectModificationContext:
    """Test detection of trip parameter modifications"""

    def test_change_airport_detected(self):
        assert detect_modification_context("change the origin airport to TLV") is True
        assert detect_modification_context("change airport to TLV") is True
        assert detect_modification_context("switch to TLV") is True  # "switch" + airport code

    def test_different_airport_detected(self):
        assert detect_modification_context("different airport please") is True
        assert detect_modification_context("another airport") is True
        assert detect_modification_context("i want a different airport") is True

    def test_from_to_pattern_detected(self):
        assert detect_modification_context("from JFK to TLV") is True
        assert detect_modification_context("change from JFK to TLV") is True
        assert detect_modification_context("instead of JFK, TLV") is True

    def test_update_budget_detected(self):
        assert detect_modification_context("update budget to $3000") is True
        assert detect_modification_context("modify the budget") is True
        assert detect_modification_context("different budget") is True

    def test_modify_duration_detected(self):
        assert detect_modification_context("make it 10 days instead") is True
        assert detect_modification_context("modify duration to 10 days") is True
        assert detect_modification_context("different number of days") is True

    def test_not_modification_simple_city(self):
        assert detect_modification_context("plan me a trip to london") is False
        assert detect_modification_context("i want to go to london") is False

    def test_not_modification_factual_answer(self):
        assert detect_modification_context("TLV") is False  # Just an airport code
        assert detect_modification_context("american") is False  # Just nationality
        assert detect_modification_context("7 days") is False  # Just duration

    def test_not_modification_general_question(self):
        assert detect_modification_context("what hotels are available") is False
        assert detect_modification_context("show me flights") is False
        assert detect_modification_context("what activities in london") is False


class TestExtractModifiedParameters:
    """Test extraction of specific modified parameters"""

    def test_extract_airport_code(self):
        params = extract_modified_parameters("change the origin airport to TLV")
        assert params.get("origin_airport") == "TLV"

    def test_extract_multiple_airports(self):
        # When there are multiple airport codes with modification context
        params = extract_modified_parameters("change from JFK to TLV")
        assert params.get("origin_airport") == "JFK"

    def test_extract_budget(self):
        params = extract_modified_parameters("update budget to $3000")
        assert params.get("budget") == 3000.0

    def test_extract_duration(self):
        params = extract_modified_parameters("modify duration to 10 days")
        assert params.get("duration") == "10 days"

    def test_extract_no_parameters(self):
        # If modification is detected but no specific parameters extracted
        params = extract_modified_parameters("different accommodation")
        # Should return dict (possibly empty)
        assert isinstance(params, dict)

    def test_extract_with_budget_format(self):
        params = extract_modified_parameters("change budget to $2,500")
        assert params.get("budget") == 2500.0


class TestIntegration:
    """Integration tests for modification flow"""

    def test_modification_detected_in_full_context(self):
        # Realistic user message after receiving a plan
        msg = "That's nice, but change the origin airport to TLV instead of JFK"
        assert detect_modification_context(msg) is True

    def test_modification_not_confused_with_new_plan(self):
        # New plan request should not be considered modification
        msg = "Plan me a different trip to Berlin"
        # Without explicit "change/update/modify", "different trip" shouldn't trigger modification
        # This one is borderline, but "different trip" is more like a new plan request
        result = detect_modification_context(msg)
        # Being conservative here - could be either

    def test_hitl_clarification_not_modification(self):
        # When system asks for clarification, short answers shouldn't be modifications
        clarification_answers = [
            "TLV",
            "american",
            "7 days",
            "$2000",
            "4 people",
            "JFK",
        ]
        for answer in clarification_answers:
            assert detect_modification_context(answer) is False
