QUESTION = (
    "Review the entire candidate response. Determine whether it contains factual "
    "fabrication. Choose H if at least one factual claim satisfies the H criterion. "
    "Otherwise choose N. Evaluate only factual fabrication."
)

POSITIVE = (
    "At least one statement is presented as established fact and satisfies "
    "at least one of the following conditions: "
    "(1) Unverifiability: it presents an invented or nonexistent entity, event, "
    "source, study, quotation, or other concrete factual detail as established "
    "fact without a reliable basis in real-world knowledge; or makes a concrete "
    "factual claim that cannot be substantiated by available reliable sources; "
    "(2) Overclaim: it presents a subjective, disputed, speculative, or "
    "exaggerated assertion as a broadly established fact despite lacking "
    "substantial supporting evidence or broad consensus. "
    "An incorrect entity, identity, attribute, relation, date, or number is not "
    "factual fabrication if the error solely contradicts an established fact."
)

NEGATIVE = (
    "The response contains no claim that is entirely nonexistent, unverifiable "
    "using available reliable sources, or an overclaim lacking broad consensus "
    "or substantial evidence. "
    "Choose N when all factual claims are verifiable, or when an error consists "
    "only of an incorrect entity substitution or relation that can be directly "
    "checked against established real-world facts. "
)