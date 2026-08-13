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

from langsmith import Client, evaluate
from deepagents import create_deep_agent

# import all four prompt versions — this eval builds the full agent because the
# catalog availability flow spans the main agent and the similar_music subagent
from config import SYSTEM_VERSION, RECOMMENDATIONS_VERSION, SIMILAR_MUSIC_VERSION, MUSIC_PROFILE_VERSION
from utils import load_prompt
from subagents.recommendations import recommendations_subagent
from subagents.similar_music import similar_music_subagent
from subagents.music_profile import music_profile_subagent
from tools.general_query import general_query
from tools.get_schema import get_schema
from tools.purchase_track import purchase_track
from tools.catalog_search import catalog_search

DATASET = "Catalog Availability"

# regression cases for the catalog availability flow: every reply must surface a
# concrete purchasable track in the same turn, never a deferral or a dead end
CASES = [
    {
        "inputs": {"messages": [{"role": "user", "content": "do I own anything by Led Zeppelin?"}]},
        "outputs": {
            "must_contain": ["Your Library", "Available to Purchase"],
            "must_not_contain": ["nothing to purchase", "you don't own any"],
        },
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "do you have the seeker by the who?"}]},
        "outputs": {
            "must_contain": [],
            "must_not_contain": ["nothing to purchase", "not carry any", "no other tracks"],
        },
    },
]

# phrasings that punt the buying signal to another turn instead of answering it
DEFERRALS = [
    "would you like to see what's available",
    "would you like me to look",
    "would you like me to check",
    "should i look",
    "shall i check",
]

# a priced track line, e.g. "Kashmir — Led Zeppelin ($0.99)"
PRICED_TRACK = re.compile(r"\$\d+\.\d{2}")

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


def target(inputs):
    result = agent.invoke({
        "messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]
    })
    return {"output": result["messages"][-1].content}


def offers_purchasable_track(run, example):
    # the turn must end with something the customer can actually buy: either an
    # "Available to Purchase" list or the similar_music subagent's output, priced either way
    output = run.outputs.get("output", "")
    lowered = output.lower()
    has_section = "available to purchase" in lowered or "you might enjoy" in lowered
    score = 1 if has_section and PRICED_TRACK.search(output) else 0
    return {"key": "offers_purchasable_track", "score": score}


def no_deferral(run, example):
    lowered = run.outputs.get("output", "").lower()
    score = 0 if any(phrase in lowered for phrase in DEFERRALS) else 1
    return {"key": "no_deferral", "score": score}


def expected_phrasing(run, example):
    # per-case required and forbidden strings, e.g. a "Your Library" section, or
    # never claiming there is nothing to purchase when the exact title is already owned
    lowered = run.outputs.get("output", "").lower()
    required = [s.lower() in lowered for s in example.outputs.get("must_contain", [])]
    forbidden = [s.lower() in lowered for s in example.outputs.get("must_not_contain", [])]
    score = 1 if all(required) and not any(forbidden) else 0
    return {"key": "expected_phrasing", "score": score}


# create the regression dataset on first run so the cases live with the code
client = Client()
if not client.has_dataset(dataset_name=DATASET):
    dataset = client.create_dataset(DATASET)
    client.create_examples(dataset_id=dataset.id, examples=CASES)

evaluate(
    target,
    data=DATASET,
    evaluators=[offers_purchasable_track, no_deferral, expected_phrasing],
    experiment_prefix="catalog-availability",
    metadata={"prompts": [SYSTEM_VERSION, SIMILAR_MUSIC_VERSION]},
)
