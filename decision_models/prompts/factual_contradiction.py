from __future__ import annotations


QUESTION = (
    "Review the entire candidate response. Determine whether it contains factual "
    "contradiction. Choose H if at least one factual claim satisfies the H criterion. "
    "Otherwise choose N. Evaluate only factual contradiction."
)

POSITIVE = (
    "At least one factual claim conflicts with reliable, verifiable real-world information. "
    "This includes an incorrect entity, relation, date, number, location, attribute, event, "
    "or other factual statement. Reliable evidence must establish that the claim is false."
)

NEGATIVE = (
    "No factual claim conflicts with reliable, verifiable real-world information. A claim "
    "that is merely unsupported, uncertain, subjective, or difficult to verify does not "
    "count as factual contradiction."
)
