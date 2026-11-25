"""
Grounding and abstain layer for AI Memory System.
Provides functions to format retrieved facts with confidence and determine if answer is grounded or should abstain.
"""

import math
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Fact:
    """Represents a retrieved fact."""
    fact_id: str
    text: str
    source: str
    confidence: float  # 0.0 to 1.0
    status: str        # active, superseded, retracted
    superseded_by: Optional[str] = None

    def __post_init__(self):
        # Validate fact_id
        if not self.fact_id or not isinstance(self.fact_id, str):
            raise ValueError("fact_id must be a non-empty string")
        if len(self.fact_id) > 256:
            raise ValueError("fact_id must not exceed 256 characters")
        # Validate text
        if not self.text or not isinstance(self.text, str):
            raise ValueError("text must be a non-empty string")
        if len(self.text) > 10000:
            raise ValueError("text must not exceed 10,000 characters")
        # Validate source
        if not self.source or not isinstance(self.source, str):
            raise ValueError("source must be a non-empty string")
        if len(self.source) > 256:
            raise ValueError("source must not exceed 256 characters")
        # Validate confidence
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a number")
        if math.isnan(self.confidence) or math.isinf(self.confidence):
            raise ValueError("confidence must be a finite number")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0 inclusive")
        # Validate status
        allowed_statuses = {"active", "superseded", "retracted"}
        if self.status not in allowed_statuses:
            raise ValueError(f"status must be one of {allowed_statuses}")
        # Validate superseded_by (if present)
        if self.superseded_by is not None:
            if not isinstance(self.superseded_by, str):
                raise ValueError("superseded_by must be a string or None")
            if not self.superseded_by:
                raise ValueError("superseded_by must not be empty if provided")
            if len(self.superseded_by) > 256:
                raise ValueError("superseded_by must not exceed 256 characters")

def format_grounded_response(facts: List[Fact], query: str, 
                            confidence_threshold: float = 0.5,
                            abstain_on_low_confidence: bool = True) -> dict:
    """
    Format a response based on retrieved facts.
    
    Args:
        facts: List of retrieved facts, ordered by relevance (highest first).
        query: The original query (for context, not used in this simple version).
        confidence_threshold: Minimum confidence to consider grounding.
        abstain_on_low_confidence: If True, abstain when confidence is below threshold.
    
    Returns:
        Dictionary with keys:
            - grounded_or_abstain: "grounded" or "abstain"
            - fact_id: ID of the top fact if grounded, else None
            - source: Source of the top fact if grounded, else None
            - confidence: Confidence of the top fact if grounded, else None
            - status: Status of the top fact if grounded, else None
            - historical_warning: True if the top fact is superseded or retracted, else False
            - fact_text: Text of the fact if grounded and active, else None
            - message: Formatted message to present to user.
    """
    # Validate confidence_threshold
    if not isinstance(confidence_threshold, (int, float)):
        raise ValueError("confidence_threshold must be a number")
    if math.isnan(confidence_threshold) or math.isinf(confidence_threshold):
        raise ValueError("confidence_threshold must be a finite number")
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold must be between 0.0 and 1.0 inclusive")
    
    if not facts:
        return {
            "grounded_or_abstain": "abstain",
            "fact_id": None,
            "source": None,
            "confidence": None,
            "status": None,
            "historical_warning": False,
            "fact_text": None,
            "message": "unverified: no relevant facts found."
        }
    
    top_fact = facts[0]
    
    # Check status for superseded/retracted - abstain for current answer
    if top_fact.status in ("superseded", "retracted"):
        # Build historical warning message
        message_parts = [
            f"[{top_fact.fact_id}]",
            f"Source: {top_fact.source}",
            f"Confidence: {top_fact.confidence:.2f}",
            f"Status: {top_fact.status}"
        ]
        if top_fact.superseded_by:
            message_parts.append(f"Superseded by: {top_fact.superseded_by}")
        base_message = " | ".join(message_parts)
        message = f"unverified: historical fact ({top_fact.status}). {base_message}"
        return {
            "grounded_or_abstain": "abstain",
            "fact_id": top_fact.fact_id,
            "source": top_fact.source,
            "confidence": top_fact.confidence,
            "status": top_fact.status,
            "historical_warning": True,
            "fact_text": None,
            "message": message
        }
    
    # Check confidence
    if abstain_on_low_confidence and top_fact.confidence < confidence_threshold:
        return {
            "grounded_or_abstain": "abstain",
            "fact_id": top_fact.fact_id,
            "source": top_fact.source,
            "confidence": top_fact.confidence,
            "status": top_fact.status,
            "historical_warning": top_fact.status in ("superseded", "retracted"),
            "fact_text": None,
            "message": f"unverified: low confidence ({top_fact.confidence:.2f})."
        }
    
    # Active fact with sufficient confidence - grounded
    message_parts = [
        f"[{top_fact.fact_id}]",
        f"Source: {top_fact.source}",
        f"Confidence: {top_fact.confidence:.2f}",
        f"Status: {top_fact.status}",
        f"{top_fact.text}"
    ]
    if top_fact.superseded_by:
        message_parts.append(f"Superseded by: {top_fact.superseded_by}")
    message = "grounded: " + " | ".join(message_parts)
    
    return {
        "grounded_or_abstain": "grounded",
        "fact_id": top_fact.fact_id,
        "source": top_fact.source,
        "confidence": top_fact.confidence,
        "status": top_fact.status,
        "historical_warning": False,
        "fact_text": top_fact.text,
        "message": message
    }

def get_instruction_block() -> str:
    """
    Returns the instruction block to be pasted into global CLI instructions.
    """
    return '''GROUNDING INSTRUCTIONS:
When answering questions, you MUST use the memory_recall tool to retrieve relevant facts.
Your answer MUST be based solely on the retrieved facts.
Format your answer as:
[FACT_ID] Source: <source> | Confidence: <confidence> | Status: <status> | <fact_text>
If the fact is superseded or retracted, the answer will be unverified and include a warning that this is a historical fact.
If no facts are retrieved or confidence is low (below 0.5), respond with:
unverified: <reason>
Do not add any information that is not present in the retrieved facts.
'''
