"""Deprecated location. The benchmark cases moved to ``evals/cases.py``
where they are typed Pydantic models. This module is retained as an
import alias so external callers (and the README's "How to add a new
domain module" section) keep working."""

from ..cases import CASES, EvalCase, ExpectedOutcome  # noqa: F401

__all__ = ["CASES", "EvalCase", "ExpectedOutcome"]
