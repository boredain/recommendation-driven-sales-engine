from deepagents import create_deep_agent

from config import SYSTEM_VERSION, SIMILAR_MUSIC_VERSION, RECOMMENDATIONS_VERSION, MUSIC_PROFILE_VERSION
from utils import load_prompt
from tools.general_query import general_query
from tools.get_schema import get_schema
from tools.purchase_track import purchase_track
from tools.catalog_search import catalog_search
from subagents.music_profile import music_profile_subagent
from subagents.recommendations import recommendations_subagent
from subagents.similar_music import similar_music_subagent


def make_graph():
    return create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        system_prompt=load_prompt(SYSTEM_VERSION),
        tools=[general_query, get_schema, purchase_track, catalog_search],
        subagents=[
            {**similar_music_subagent, "system_prompt": load_prompt(SIMILAR_MUSIC_VERSION)},
            {**recommendations_subagent, "system_prompt": load_prompt(RECOMMENDATIONS_VERSION)},
            {**music_profile_subagent, "system_prompt": load_prompt(MUSIC_PROFILE_VERSION)},
        ],
    )