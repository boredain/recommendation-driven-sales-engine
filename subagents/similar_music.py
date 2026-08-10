similar_music_subagent = {
    "name": "similar_music",
    "description": (
        "Recommends 5 similar tracks from the catalog when the customer's searched "
        "artist, album, genre, or track is not available. Pass the customer's original "
        "search query in the task so the subagent can reason about the closest matching "
        "genre and find unowned tracks in that genre."
    ),
    "model": "anthropic:claude-haiku-4-5-20251001",
}

_runner = None
_running = False


def run_similar_music(search_query: str) -> str:
    """Invoke the similar_music subagent directly and return its output verbatim."""
    global _runner, _running
    # the subagent itself calls general_query, whose zero-unowned-results branch calls
    # back into here — the guard stops that from recursing
    if _running:
        return ""

    from deepagents import create_deep_agent

    from config import SIMILAR_MUSIC_VERSION
    from utils import load_prompt
    from tools.catalog_search import catalog_search
    from tools.general_query import general_query
    from tools.get_schema import get_schema

    if _runner is None:
        _runner = create_deep_agent(
            model=similar_music_subagent["model"],
            system_prompt=load_prompt(SIMILAR_MUSIC_VERSION),
            tools=[general_query, get_schema, catalog_search],
        )

    _running = True
    try:
        result = _runner.invoke({"messages": [{"role": "user", "content": search_query}]})
    finally:
        _running = False
    return result["messages"][-1].content
