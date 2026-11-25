"""
Evaluation adapter for measuring unsupported-claim and abstain precision.
This module provides functions to compute metrics but does not run the evaluation.
"""

from typing import List, Dict

def compute_unsupported_claim_rate(supported_labels: List[bool]) -> float:
    """
    Compute the rate of unsupported claims at claim level.
    An unsupported claim is when the system says "grounded" but the fact does not support the answer.
    
    Args:
        supported_labels: List of boolean labels from independent evaluators indicating 
                         whether each claim is supported by the retrieved facts.
                         
    Returns:
        Unsupported claim rate as a float between 0 and 1.
        Returns 0.0 if the input list is empty.
    """
    for i, label in enumerate(supported_labels):
        if type(label) is not bool:
            raise ValueError(f"supported_labels[{i}] must be a bool")

    if not supported_labels:
        return 0.0
    
    unsupported = sum(1 for label in supported_labels if not label)
    total = len(supported_labels)
    return unsupported / total


def compute_unsupported_claim_rate_old(
    predictions: List[Dict],
    ground_truth: List[List[str]]
) -> float:
    """
    Compute the rate of unsupported claims.
    An unsupported claim is when the system says "grounded" but the fact does not support the answer.
    For simplicity, we assume that if the system says "grounded", we check if the fact ID 
    is in the list of fact IDs that are considered to support the answer (provided in ground_truth).
    
    Args:
        predictions: List of prediction dicts from format_grounded_response.
        ground_truth: List of lists of fact IDs that support the answer for each query.
    
    Returns:
        Unsupported claim rate as a float between 0 and 1.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(f"Length mismatch: predictions ({len(predictions)}) != ground_truth ({len(ground_truth)})")
    
    if not predictions:
        return 0.0
    
    unsupported = 0
    total_grounded = 0
    for pred, supportive_set in zip(predictions, ground_truth):
        if pred["grounded_or_abstain"] == "grounded":
            total_grounded += 1
            if pred["fact_id"] not in supportive_set:
                unsupported += 1
    
    if total_grounded == 0:
        return 0.0
    return unsupported / total_grounded

def compute_abstain_precision(
    predictions: List[Dict],
    ground_truth: List[bool]
) -> float:
    """
    Compute the precision of abstention.
    That is, when the system abstains, how often is it correct to do so?
    
    Args:
        predictions: List of prediction dicts from format_grounded_response.
        ground_truth: List of booleans indicating whether abstention was correct (True) or not (False).
    
    Returns:
        Abstain precision as a float between 0 and 1.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(f"Length mismatch: predictions ({len(predictions)}) != ground_truth ({len(ground_truth)})")
    
    for i, truth in enumerate(ground_truth):
        if not isinstance(truth, bool):
            raise ValueError(f"ground_truth[{i}] is not a bool: {truth!r}")
    
    true_positives = 0  # correctly abstained
    false_positives = 0  # incorrectly abstained (should have answered)
    
    for pred, truth in zip(predictions, ground_truth):
        if pred["grounded_or_abstain"] == "abstain":
            if truth:
                true_positives += 1
            else:
                false_positives += 1
    
    if true_positives + false_positives == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)

def compute_citation_support_rate(predictions: List[Dict], ground_truth_support: List[List[str]]) -> float:
    """
    Compute the rate of citation support (retrieval citation membership).
    This measures whether the cited fact IDs are in the ground truth support set.
    
    Args:
        predictions: List of prediction dicts from format_grounded_response.
        ground_truth_support: List of lists of fact IDs that support the answer for each query.
                              Each inner list corresponds to the prediction at the same index.
    
    Returns:
        Citation support rate as a float between 0 and 1.
        Returns 0.0 if inputs are empty or if no predictions are grounded.
    """
    if len(predictions) != len(ground_truth_support):
        raise ValueError(f"Length mismatch: predictions ({len(predictions)}) != ground_truth_support ({len(ground_truth_support)})")
    
    if not predictions:
        return 0.0
    
    correct_citations = 0
    total_grounded = 0
    
    for pred, supportive_set in zip(predictions, ground_truth_support):
        if pred["grounded_or_abstain"] == "grounded":
            total_grounded += 1
            if pred["fact_id"] in supportive_set:
                correct_citations += 1
    
    if total_grounded == 0:
        return 0.0
    return correct_citations / total_grounded

if __name__ == "__main__":
    # Example usage (not to be run in production)
    print("Evaluation adapter for grounding layer.")
    print("This module provides functions to compute metrics but does not run the evaluation.")
