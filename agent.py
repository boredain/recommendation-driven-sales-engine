from deepagents import create_deep_agent

from config import (
    SYSTEM_VERSION,
    SIMILAR_MUSIC_VERSION,
    RECOMMENDATIONS_VERSION,
    MUSIC_PROFILE_VERSION,
    DEMO_CUSTOMER_ID,
    ENVIRONMENT,
    REVISION_ID,
)
from utils import load_prompt
from tools.general_query import general_query
from tools.get_schema import get_schema
from tools.purchase_track import purchase_track
from tools.catalog_search import catalog_search
from subagents.music_profile import music_profile_subagent
from subagents.recommendations import recommendations_subagent
from subagents.similar_music import similar_music_subagent


def trace_metadata():
    """Root-run metadata for attributing traces to a prompt version, environment and customer."""
    return {
        "system_version": SYSTEM_VERSION,
        "recommendations_version": RECOMMENDATIONS_VERSION,
        "similar_music_version": SIMILAR_MUSIC_VERSION,
        "music_profile_version": MUSIC_PROFILE_VERSION,
        "environment": ENVIRONMENT,
        "revision_id": REVISION_ID,
        # Identity stays in metadata only -- it must never reach customer-facing output text.
        "user_id": str(DEMO_CUSTOMER_ID),
        "customer_id": str(DEMO_CUSTOMER_ID),
    }


def make_graph():
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        system_prompt=load_prompt(SYSTEM_VERSION),
        tools=[general_query, get_schema, purchase_track, catalog_search],
        subagents=[
            {**similar_music_subagent, "system_prompt": load_prompt(SIMILAR_MUSIC_VERSION)},
            {**recommendations_subagent, "system_prompt": load_prompt(RECOMMENDATIONS_VERSION)},
            {**music_profile_subagent, "system_prompt": load_prompt(MUSIC_PROFILE_VERSION)},
        ],
    )
    return agent.with_config({"metadata": trace_metadata()})