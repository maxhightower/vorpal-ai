# Using the harness as an agent tool

The harness exposes one method designed for agent tool-use:
``FactualityPipeline.verify_draft(...)``. It takes a pre-written draft
answer (the agent's own composition) and returns a ``FinalAnswer`` with
the draft's unsupported claims dropped or qualified.

This is **private infrastructure** — there is no public service. The
wrapper is callable only by code that imports the module or hits the
FastAPI server you run on your own infrastructure. Access control is
your responsibility.

## When to use it

Wire `verify_draft` as a tool when you have an outer agent that:

- Composes answers itself (text generation is the agent's job).
- Needs each claim it makes to be auditable / verifiable.
- Should drop or qualify any assertion the harness can't verify.

This is the **fact-check** pattern: agent is fluent, harness keeps it
honest.

## Python entry point

```python
from factuality_harness.interfaces.factory import build_pipeline
from factuality_harness.infrastructure.retrieval.base import Document

pipeline = build_pipeline()

result = pipeline.verify_draft(
    draft="The percentage increase from 100 to 125 is 25%, which proves the campaign worked.",
    question="What is the percentage increase from 100 to 125, and did the campaign cause the growth?",
    documents=[],          # optional Document objects
    extra_context={},      # optional structured context (sql, experiment data, etc.)
)

# result.answer                            -> revised draft text
# result.unsupported_or_uncertain_claims   -> list of claims dropped/qualified
# result.confidence_summary                -> "COMPUTED: 1, UNSUPPORTED: 1"
# result.audit_id                          -> handle to full provenance trace
# result.evidence_table                    -> the underlying evidence + verdicts
```

## CLI entry point

```sh
fh verify "The percentage increase from 100 to 125 is 25%, which proves the campaign worked." \
    --question "What is the percentage increase, and did the campaign cause the growth?"
```

## Anthropic tool-use schema

To call from inside an Anthropic Messages API request, declare the tool
in `tools` and execute `pipeline.verify_draft(**tool.input)` in your
agent loop. The recommended schema:

```python
verify_draft_tool = {
    "name": "verify_draft",
    "description": (
        "Fact-check a pre-written draft answer against deterministic "
        "verification tools. Returns the draft with unsupported claims "
        "dropped or qualified, plus a list of what was removed and an "
        "audit ID for provenance. Use after composing any answer that "
        "makes factual, numerical, causal, predictive, procedural, or "
        "optimization claims."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "draft": {
                "type": "string",
                "description": "The draft answer text to verify.",
            },
            "question": {
                "type": "string",
                "description": (
                    "Original user question that the draft answers. "
                    "Drives evidence gathering. Optional but recommended."
                ),
            },
            "documents": {
                "type": "array",
                "description": (
                    "Optional documents the harness should retrieve from "
                    "(policy text, reference material, etc.)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "text": {"type": "string"},
                        "effective_date": {"type": "string", "format": "date"},
                    },
                    "required": ["name", "text"],
                },
            },
        },
        "required": ["draft"],
    },
}
```

### Agent loop wiring (Python, Anthropic SDK)

```python
import anthropic
from factuality_harness.interfaces.factory import build_pipeline
from factuality_harness.infrastructure.retrieval.base import Document

client = anthropic.Anthropic()
pipeline = build_pipeline()

response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=16000,
    tools=[verify_draft_tool],
    messages=[{"role": "user", "content": user_question}],
)

for block in response.content:
    if block.type == "tool_use" and block.name == "verify_draft":
        docs = [Document.model_validate(d) for d in block.input.get("documents", [])]
        result = pipeline.verify_draft(
            draft=block.input["draft"],
            question=block.input.get("question"),
            documents=docs,
        )
        # Hand result back as a tool_result content block, slimmed for
        # context efficiency:
        tool_result = {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result.model_dump_json(
                include={
                    "answer", "confidence_summary",
                    "unsupported_or_uncertain_claims", "audit_id",
                }
            ),
        }
```

## What to send back to the agent

The full `FinalAnswer` includes the entire `EvidenceTable`, which can be
verbose. For agent context-budget reasons, the recommended slim payload
is:

| Field | Why it's worth sending back |
|---|---|
| `answer` | The revised draft (this is the actionable output). |
| `confidence_summary` | Short string the agent can quote. |
| `unsupported_or_uncertain_claims` | List of what was dropped or qualified — the agent should mention these. |
| `audit_id` | So the agent can offer "see audit X for provenance" to the user. |

If the agent needs full provenance later, it can call a separate
`get_audit(audit_id)` tool that returns the `AuditTrace`.

## Trust model

- The harness is **not a sandbox**. Anything the agent passes to
  `verify_draft` is processed in your own Python process with your
  configured tools. Don't expose this to the public internet.
- Tools the harness invokes (SQL, code execution, retrieval) inherit
  whatever credentials the harness has. Audit your `tools` config the
  same way you'd audit any other code path.
- The `verify_draft` method is read-only with respect to your data
  except where the underlying tools have side effects (writing to
  audit storage, hitting paid APIs, etc.). Same caveats as `run()`.
