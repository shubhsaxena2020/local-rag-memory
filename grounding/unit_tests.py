"""
Deterministic unit tests for the grounding layer.
These tests are not to be run in production but are included for verification.
"""

from grounding import format_grounded_response, Fact

def test_no_facts():
    """Test abstention when no facts are retrieved."""
    result = format_grounded_response([], "test query")
    assert result["grounded_or_abstain"] == "abstain"
    assert result["message"] == "unverified: no relevant facts found."
    assert result["fact_id"] is None

def test_low_confidence():
    """Test abstention when confidence is below threshold."""
    facts = [Fact("f1", "Test fact", "test_source", 0.4, "active")]
    result = format_grounded_response(facts, "test query", confidence_threshold=0.5)
    assert result["grounded_or_abstain"] == "abstain"
    assert result["confidence"] == 0.4
    assert result["fact_id"] == "f1"

def test_grounded():
    """Test grounded response when confidence is sufficient."""
    facts = [Fact("f1", "Test fact", "test_source", 0.8, "active")]
    result = format_grounded_response(facts, "test query", confidence_threshold=0.5)
    assert result["grounded_or_abstain"] == "grounded"
    assert result["fact_id"] == "f1"
    assert result["source"] == "test_source"
    assert result["confidence"] == 0.8
    assert result["status"] == "active"
    assert result["superseded_warning"] == False

def test_superseded_fact():
    """Test that a superseded fact triggers a warning."""
    facts = [Fact("f1", "Test fact", "test_source", 0.9, "superseded", superseded_by="f2")]
    result = format_grounded_response(facts, "test query", confidence_threshold=0.5)
    assert result["grounded_or_abstain"] == "grounded"
    assert result["superseded_warning"] == True
    assert result["status"] == "superseded"

def test_abstain_flag_with_superseded_low_confidence():
    """Test abstention when a superseded fact has low confidence."""
    facts = [Fact("f1", "Test fact", "test_source", 0.4, "superseded", superseded_by="f2")]
    result = format_grounded_response(facts, "test query", confidence_threshold=0.5)
    assert result["grounded_or_abstain"] == "abstain"
    assert result["superseded_warning"] == True

if __name__ == "__main__":
    test_no_facts()
    test_low_confidence()
    test_grounded()
    test_superseded_fact()
    test_abstain_flag_with_superseded_low_confidence()
    print("All unit tests passed.")