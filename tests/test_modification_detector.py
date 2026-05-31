"""Tests for modification detection."""

import pytest
from src.utils.modification_detector import detect_modification_context, extract_modified_parameters


@pytest.mark.parametrize("msg", [
    "change the origin airport to TLV",
    "change airport to TLV",
    "switch to TLV",
    "different airport please",
    "another airport",
    "i want a different airport",
    "from JFK to TLV",
    "change from JFK to TLV",
    "instead of JFK, TLV",
    "update budget to $3000",
    "modify the budget",
    "different budget",
    "make it 10 days instead",
    "modify duration to 10 days",
    "different number of days",
    "That's nice, but change the origin airport to TLV instead of JFK",
])
def test_detect_modification_true(msg):
    assert detect_modification_context(msg) is True


@pytest.mark.parametrize("msg", [
    "plan me a trip to london",
    "i want to go to london",
    "TLV",
    "american",
    "7 days",
    "$2000",
    "4 people",
    "JFK",
    "what hotels are available",
    "show me flights",
    "what activities in london",
])
def test_detect_modification_false(msg):
    assert detect_modification_context(msg) is False


@pytest.mark.parametrize("msg,field,expected", [
    ("change the origin airport to TLV",  "origin_airport", "TLV"),
    ("change from JFK to TLV",            "origin_airport", "JFK"),
    ("update budget to $3000",            "budget",         3000.0),
    ("change budget to $2,500",           "budget",         2500.0),
    ("modify duration to 10 days",        "duration",       "10 days"),
])
def test_extract_parameters(msg, field, expected):
    params = extract_modified_parameters(msg)
    assert isinstance(params, dict)
    assert params.get(field) == expected
