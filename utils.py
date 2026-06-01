from pathlib import Path
from config import DEMO_CUSTOMER_ID


def load_prompt(name: str) -> str:
    text = (Path(__file__).parent / "prompts" / f"{name}.txt").read_text()
    return text.format(customer_id=DEMO_CUSTOMER_ID)