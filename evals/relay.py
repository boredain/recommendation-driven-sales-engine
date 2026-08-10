import re
import sys
from pathlib import Path

# add project root to Python path so imports like "from config import ..." work
# when this script is run directly from the evals/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
# load ANTHROPIC_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING from .env
# must happen before any LangSmith or Anthropic calls
load_dotenv()

from langchain_core.messages import AIMessage, ToolMessage
from langsmith import evaluate
from deepagents import create_deep_agent

from config import SYSTEM_VERSION, RECOMMENDATIONS_VERSION, SIMILAR_MUSIC_VERSION, MUSIC_PROFILE_VERSION
from utils import load_prompt
from subagents.recommendations import build_recommendations_subagent
from subagents.similar_music import similar_music_subagent
from subagents.music_profile import music_profile_subagent
from tools.general_query import general_query
from tools.get_schema import get_schema
from tools.purchase_track import purchase_track
from tools.catalog_search import catalog_search

# the recommendations subagent always returns its top 5 tracks
EXPECTED_RECOMMENDATIONS = 5

# the collapsed one-liner shape observed in production: "1. **Track** – Artist | Genre | $0.99"
CONDENSED_ENTRY = re.compile(r"[–-]\s*[^|\n]+\|[^|\n]+\|\s*\$")

# allow overriding the main agent system prompt version via CLI: python evals/relay.py system/system_v2
version = sys.argv[1] if len(sys.argv) > 1 else SYSTEM_VERSION

# build the full agent — identical to agent.py — because this eval tests what the
# orchestrator relays to the customer, not what the subagent produces in isolation
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    system_prompt=load_prompt(version),
    tools=[general_query, get_schema, purchase_track, catalog_search],
    subagents=[
        {**similar_music_subagent, "system_prompt": load_prompt(SIMILAR_MUSIC_VERSION)},
        build_recommendations_subagent(
            load_prompt(RECOMMENDATIONS_VERSION),
            [general_query, get_schema, catalog_search],
        ),
        {**music_profile_subagent, "system_prompt": load_prompt(MUSIC_PROFILE_VERSION)},
    ],
)


def count_lines(text, label):
    # counts how many lines carry a given field label, e.g. "Why:" or "Genre:"
    return sum(1 for line in text.splitlines() if label in line)


def target(inputs):
    # called by LangSmith for each dataset row; invokes the full agent and captures both
    # what the recommendations subagent returned and how the orchestrator relayed it
    result = agent.invoke({
        "messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]
    })
    messages = result["messages"]

    # collect tool call ids for delegations to the recommendations subagent so the
    # matching ToolMessage (the subagent's output) can be found below
    task_ids = set()
    for msg in messages:
        for tc in (getattr(msg, "tool_calls", None) or []):
            if tc["name"] == "task" and tc["args"].get("subagent_type") == "recommendations":
                task_ids.add(tc["id"])

    subagent_output = ""
    relay_message = ""
    for index, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and msg.tool_call_id in task_ids:
            subagent_output = msg.content
            # the AI message immediately after the subagent output is the relay to the customer
            for later in messages[index + 1:]:
                if isinstance(later, AIMessage) and later.text.strip():
                    relay_message = later.text
                    break

    return {"subagent_output": subagent_output, "relay_message": relay_message}


def why_lines_relayed(run, example):
    # general invariant: if the subagent returned N Why lines, the relaying AI message
    # must contain N Why lines — a dropped Why line means the personalization was lost
    subagent_output = run.outputs.get("subagent_output", "")
    relay_message = run.outputs.get("relay_message", "")
    expected = count_lines(subagent_output, "Why:")
    actual = count_lines(relay_message, "Why:")
    return {"key": "why_lines_relayed", "score": 1 if expected and expected == actual else 0}


def entry_fields_present(run, example):
    # every relayed entry keeps its own Genre and Why line
    relay_message = run.outputs.get("relay_message", "")
    why = count_lines(relay_message, "Why:")
    genre = count_lines(relay_message, "Genre:")
    passed = why == EXPECTED_RECOMMENDATIONS and genre == EXPECTED_RECOMMENDATIONS
    return {"key": "entry_fields_present", "score": 1 if passed else 0}


def no_condensed_entries(run, example):
    # scores 0 if the orchestrator collapsed a multi-line entry onto one line
    relay_message = run.outputs.get("relay_message", "")
    return {"key": "no_condensed_entries", "score": 0 if CONDENSED_ENTRY.search(relay_message) else 1}


# run against the existing LangSmith "Recommendation" dataset — the same discovery
# requests ("what should I listen to next"), but scored on the orchestrator's relay
evaluate(
    target,
    data="Recommendation",
    evaluators=[why_lines_relayed, entry_fields_present, no_condensed_entries],
    experiment_prefix="relay",
    metadata={"prompts": [version]},
)
