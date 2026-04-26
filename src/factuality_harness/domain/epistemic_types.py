from enum import Enum


class EpistemicType(str, Enum):
    """The kind of truth a claim requires.

    Drives routing: each type maps to a verification pathway in the router.
    """

    DIRECT_FACT = "DIRECT_FACT"
    NUMERICAL = "NUMERICAL"
    LOGICAL = "LOGICAL"
    PROCEDURAL = "PROCEDURAL"
    CAUSAL = "CAUSAL"
    PREDICTIVE = "PREDICTIVE"
    OPTIMIZATION = "OPTIMIZATION"
    INTERPRETIVE = "INTERPRETIVE"
    SPECULATIVE = "SPECULATIVE"
    UNKNOWN = "UNKNOWN"
