from config import MUSIC_PROFILE_VERSION
from utils import load_prompt

music_profile_subagent = {
    "name": "music_profile",
    "description": (
        "Generates a Spotify Wrapped-style music profile for the customer based on their "
        "full purchase history. Call this when the customer asks for their music profile, "
        "taste summary, listening history, or a Wrapped-style report."
    ),
    "system_prompt": load_prompt(MUSIC_PROFILE_VERSION),
    "model": "anthropic:claude-haiku-4-5-20251001",
}