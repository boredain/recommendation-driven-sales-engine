from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

recommendations_subagent = {
    "name": "recommendations",
    "description": (
        "Generates personalized track recommendations based on the customer's purchase history. "
        "Call this when the customer asks for recommendations, suggestions, what to listen to "
        "next, or anything resembling music discovery."
    ),
    "model": "anthropic:claude-haiku-4-5-20251001",
}


class Recommendation(BaseModel):
    track: str = Field(description="Track name exactly as it appears in the catalog")
    artist: str = Field(description="Artist name exactly as it appears in the catalog")
    genre: str = Field(description="Genre of the track")
    price: float = Field(description="Unit price in dollars, without a currency symbol")
    why: str = Field(description="One personalized sentence explaining the signal behind this recommendation")


class RecommendationList(BaseModel):
    recommendations: list[Recommendation]


def render_recommendations(recommendations: list[Recommendation]) -> str:
    """Render recommendations as the customer-facing markdown list."""
    lines = []
    for position, rec in enumerate(recommendations, start=1):
        lines.append(f"{position}. **{rec.track}** — {rec.artist}")
        lines.append(f"   Genre: {rec.genre} | Price: ${rec.price:.2f}")
        lines.append(f"   Why: {rec.why}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_recommendations_subagent(system_prompt: str, tools: list) -> dict:
    """Compile the recommendations subagent so its list is rendered in code, not copied by the orchestrator."""
    agent = create_agent(
        model=recommendations_subagent["model"],
        system_prompt=system_prompt,
        tools=tools,
        response_format=RecommendationList,
    )

    def run(state: dict) -> dict:
        result = agent.invoke({"messages": state["messages"]})
        structured = result.get("structured_response")
        if structured is None:
            return {"messages": [result["messages"][-1]]}
        return {"messages": [AIMessage(render_recommendations(structured.recommendations))]}

    return {
        "name": recommendations_subagent["name"],
        "description": recommendations_subagent["description"],
        "runnable": RunnableLambda(run, name=recommendations_subagent["name"]),
    }
