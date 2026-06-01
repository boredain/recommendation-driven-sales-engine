import sys
from pathlib import Path

# add project root to Python path so imports like "from config import ..." work
# when this script is run directly from the evals/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
# load ANTHROPIC_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING from .env
# must happen before any LangSmith or Anthropic calls
load_dotenv()

from langsmith import evaluate
from deepagents import create_deep_agent

# import all four prompt versions — needed because this eval builds the full agent with all subagents
from config import SYSTEM_VERSION, RECOMMENDATIONS_VERSION, SIMILAR_MUSIC_VERSION, MUSIC_PROFILE_VERSION
from utils import load_prompt
from subagents.recommendations import recommendations_subagent
from subagents.similar_music import similar_music_subagent
from subagents.music_profile import music_profile_subagent
from tools.general_query import general_query
from tools.get_schema import get_schema
# purchase_track included — routing to the purchase flow is one of the things being tested
from tools.purchase_track import purchase_track
from tools.catalog_search import catalog_search

# allow overriding the main agent system prompt version via CLI: python evals/routing.py system/system_v2
# falls back to the default SYSTEM_VERSION in config.py if no argument is passed
version = sys.argv[1] if len(sys.argv) > 1 else SYSTEM_VERSION

# build the full agent with all tools and subagents — identical to agent.py —
# because routing eval tests the main agent's decision to call the right tool or subagent
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    system_prompt=load_prompt(version),
    tools=[general_query, get_schema, purchase_track, catalog_search],
    subagents=[
        {**similar_music_subagent, "system_prompt": load_prompt(SIMILAR_MUSIC_VERSION)},
        {**recommendations_subagent, "system_prompt": load_prompt(RECOMMENDATIONS_VERSION)},
        {**music_profile_subagent, "system_prompt": load_prompt(MUSIC_PROFILE_VERSION)},
    ],
)


def target(inputs):
    # called by LangSmith for each dataset row;
    # invokes the full agent and extracts the list of tools/subagents it called
    result = agent.invoke({
        "messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]
    })
    called = []
    for msg in result["messages"]:
        # skip messages that are not tool call messages (e.g. plain text responses)
        if not hasattr(msg, "tool_calls"):
            continue
        for tc in (msg.tool_calls or []):
            if tc["name"] == "task":
                # subagent delegations appear as a tool named "task" with a subagent_type argument;
                # extract the subagent name (e.g. "recommendations", "music_profile")
                called.append(tc["args"].get("subagent_type", ""))
            else:
                # direct tool calls (e.g. "general_query", "catalog_search") use their actual name
                called.append(tc["name"])
    # return flat list of everything the agent called, e.g. ["general_query"] or ["recommendations"]
    return {"called_tools": called}


def correct_routing(run, example):
    # scoring function called by LangSmith after target() runs for each dataset row;
    # scores 1 if the expected tool appears anywhere in the called list, 0 if not
    expected = example.outputs["expected_tool"]
    called = run.outputs.get("called_tools", [])
    return {"key": "correct_routing", "score": 1 if expected in called else 0}


# run the eval against the LangSmith "Routing" dataset;
# correct_routing scores each row; experiment tagged with prompt version for cross-version comparison
evaluate(
    target,
    data="Routing",
    evaluators=[correct_routing],
    experiment_prefix="routing",
    metadata={"prompts": [version]},
)