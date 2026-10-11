from __future__ import annotations


QUESTION = (
    "Review the entire candidate response, including the relationships among its statements, "
    "reasoning steps, calculations, and conclusion. Determine whether it is internally "
    "logically inconsistent. Choose H if it contains any internal contradiction defined by "
    "the H criterion. Otherwise choose N."
)

POSITIVE = (
    "Two or more parts of the response are mutually incompatible; the response contradicts "
    "one of its earlier statements; or its conclusion conflicts with its own premises, "
    "assumptions, calculations, or stated reasoning steps."
)

NEGATIVE = (
    "The statements, reasoning steps, calculations, and conclusion are internally consistent "
    "with one another. A factual error, unverifiable claim, lack of contextual support, or "
    "failure to follow an instruction does not by itself count as internal logical inconsistency."
)
