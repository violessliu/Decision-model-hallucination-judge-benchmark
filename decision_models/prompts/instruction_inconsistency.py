from __future__ import annotations


QUESTION = (
    "Review the entire user instruction and candidate response. Determine whether the response "
    "fails to follow the instruction. Choose H if the response violates at least one applicable "
    "explicit requirement defined by the H criterion. Otherwise choose N. Evaluate only "
    "instruction compliance."
)

POSITIVE = (
    "The response performs the wrong task, violates an explicit requirement or prohibition, "
    "omits required content or a required deliverable, or fails to satisfy an explicitly "
    "required format, language, length, style, role, structure, or other constraint."
)

NEGATIVE = (
    "The response performs the requested task and satisfies every applicable explicit requirement "
    "and constraint. Optional preferences do not have to be satisfied. A factual error, "
    "unverifiable statement, contextual error, or internal logical problem does not by itself "
    "count unless it directly violates an explicit instruction."
)
