from __future__ import annotations


QUESTION = (
    "Review the entire candidate response against the supplied context. Determine whether "
    "the response asserts at least one substantive factual claim that is inconsistent with "
    "or unsupported by that context. Choose H only if such a claim satisfies the H criterion. "
    "Otherwise choose N. Use the supplied context as the evidence source, and evaluate only "
    "contextual faithfulness rather than formatting or general instruction compliance."
)

POSITIVE = (
    "At least one substantive factual claim in the candidate response contradicts the "
    "supplied context or introduces information that cannot reasonably be supported or "
    "inferred from the supplied context. Judge faithfulness to the context even if the "
    "introduced information might be true in the real world."
)

NEGATIVE = (
    "Every substantive factual claim in the candidate response is supported by, entailed "
    "by, or reasonably inferable from the supplied context. Also choose N when the response "
    "merely omits contextual details, declines or is unable to answer, uses harmless "
    "introductory or meta wording, or expresses a reasonable paraphrase, summary, abstraction, "
    "or inference. Formatting, length, style, and other instruction-following violations do "
    "not by themselves count as context inconsistency."
)
