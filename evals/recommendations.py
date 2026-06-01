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

from config import RECOMMENDATIONS_VERSION
from utils import load_prompt
from subagents.recommendations import recommendations_subagent
from tools.general_query import general_query
from tools.get_schema import get_schema
# purchase_track excluded — purchasing is not part of the recommendations subagent
from tools.catalog_search import catalog_search

# allow overriding the prompt version via CLI: python evals/recommendations.py recommendations/recommendations_v2
# falls back to the default version in config.py if no argument is passed
version = sys.argv[1] if len(sys.argv) > 1 else RECOMMENDATIONS_VERSION

# build just the recommendations subagent in isolation — not the full agent —
# so the eval targets only the recommendations logic and failures are easier to diagnose
subagent = create_deep_agent(
    model=recommendations_subagent["model"],
    system_prompt=load_prompt(version),
    tools=[general_query, get_schema, catalog_search],
)


def target(inputs):
    # called by LangSmith for each row in the dataset;
    # invokes the subagent with the user message and returns the final response
    result = subagent.invoke({
        "messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]
    })
    # [-1] gets the last message in the conversation — always the subagent's final answer
    return {"output": result["messages"][-1].content}


# run the eval against the LangSmith "Recommendation" dataset;
# tags each experiment run with the prompt version so results can be compared across versions in LangSmith
evaluate(
    target,
    data="Recommendation",
    experiment_prefix="recommendation",
    metadata={"prompts": [version]},
)