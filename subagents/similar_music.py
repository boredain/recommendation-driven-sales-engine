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