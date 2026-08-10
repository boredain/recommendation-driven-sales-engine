import sys
from pathlib import Path

# add project root to Python path so imports like "from config import ..." work
# when this test is run directly from the tests/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from subagents.recommendations import (
    RecommendationsOutputGuardMiddleware,
    sanitize_recommendations_output,
)

# captured verbatim from a traced recommendations delegation that leaked internals
POLLUTED_OUTPUT = """# Recommendations for Customer 5

Based on the Round 3 co-occurrence analysis (Tier 1 matches, co-occurrence count 4)
and the customer's genre tallies (Rock: 12, Metal: 7):

1. **Fast As a Shark** — Accept
   Genre: Rock | Price: $0.99
   Why: You own three other Accept tracks.

2. **Restless and Wild** — Accept
   Genre: Rock | Price: $0.99
   Why: It shares playlists with tracks you already bought.

Let me know if you want more, Customer 5!"""

LEAKED_TERMS = ["Customer 5", "Tier", "Round", "co-occurrence", "CustomerId"]


def _task_request(subagent_type):
    return ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {"description": "what should I listen to next?", "subagent_type": subagent_type},
            "id": "call_1",
        },
        tool=None,
        state={},
        runtime=None,
    )


def test_sanitizer_strips_header_and_customer_id():
    cleaned = sanitize_recommendations_output(POLLUTED_OUTPUT)
    assert cleaned.startswith("1.")
    for term in LEAKED_TERMS:
        assert term not in cleaned


def test_sanitizer_leaves_compliant_output_untouched():
    compliant = "1. **Fast As a Shark** — Accept\n   Genre: Rock | Price: $0.99\n   Why: You own three other Accept tracks."
    assert sanitize_recommendations_output(compliant) == compliant


def test_middleware_sanitizes_recommendations_result():
    guard = RecommendationsOutputGuardMiddleware()
    result = guard.wrap_tool_call(
        _task_request("recommendations"),
        lambda request: Command(
            update={"messages": [ToolMessage(content=POLLUTED_OUTPUT, tool_call_id="call_1")]}
        ),
    )
    content = result.update["messages"][0].content
    assert content.startswith("1.")
    for term in LEAKED_TERMS:
        assert term not in content


def test_middleware_ignores_other_subagents():
    guard = RecommendationsOutputGuardMiddleware()
    result = guard.wrap_tool_call(
        _task_request("music_profile"),
        lambda request: Command(
            update={"messages": [ToolMessage(content=POLLUTED_OUTPUT, tool_call_id="call_1")]}
        ),
    )
    assert result.update["messages"][0].content == POLLUTED_OUTPUT


if __name__ == "__main__":
    # runnable without pytest: python tests/test_recommendations_output.py
    for name, case in sorted(list(globals().items())):
        if name.startswith("test_") and callable(case):
            case()
            print(f"ok {name}")
