import dataclasses
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from config import DEMO_CUSTOMER_ID

recommendations_subagent = {
    "name": "recommendations",
    "description": (
        "Generates personalized track recommendations based on the customer's purchase history. "
        "Call this when the customer asks for recommendations, suggestions, what to listen to "
        "next, or anything resembling music discovery."
    ),
    "model": "anthropic:claude-haiku-4-5-20251001",
}

# the subagent's output contract ("return only the numbered list") is prompt-level and
# was violated on every delegation, leaking the internal customer id and ranking internals
_FIRST_ITEM_PATTERN = re.compile(r"^\s*1\.")
_INTERNAL_ID_PATTERN = re.compile(
    rf"customer\s*#?\s*{DEMO_CUSTOMER_ID}\b|customerid", re.IGNORECASE
)


def sanitize_recommendations_output(text: str) -> str:
    """Drop anything before the first numbered item and any line naming the internal customer id."""
    lines = (text or "").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if _FIRST_ITEM_PATTERN.match(line)), 0
    )
    kept = [line for line in lines[start:] if not _INTERNAL_ID_PATTERN.search(line)]
    return "\n".join(kept).strip()


def _sanitize_message(message: Any) -> Any:
    if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
        return message
    return message.model_copy(
        update={"content": sanitize_recommendations_output(message.content)}
    )


def _sanitize_result(
    request: ToolCallRequest, result: ToolMessage | Command
) -> ToolMessage | Command:
    tool_call = request.tool_call
    if tool_call["name"] != "task":
        return result
    if tool_call.get("args", {}).get("subagent_type") != recommendations_subagent["name"]:
        return result
    if isinstance(result, Command):
        if not isinstance(result.update, dict) or "messages" not in result.update:
            return result
        return dataclasses.replace(
            result,
            update={
                **result.update,
                "messages": [_sanitize_message(m) for m in result.update["messages"]],
            },
        )
    return _sanitize_message(result)


class RecommendationsOutputGuardMiddleware(AgentMiddleware):
    """Enforces the recommendations subagent output contract before the main agent relays it."""

    name = "RecommendationsOutputGuard"

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        return _sanitize_result(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        return _sanitize_result(request, await handler(request))
