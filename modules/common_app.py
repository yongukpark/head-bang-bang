from pathlib import Path


APP_NAME = "HeadScope"
ROOT_DIR = Path(__file__).resolve().parents[1]
SAVED_PROMPTS_DIR = ROOT_DIR / "saved_prompts"
PROMPT_LIBRARY_FILE = SAVED_PROMPTS_DIR / "prompt_library.json"
SAVED_HEADS_DIR = ROOT_DIR / "saved_heads"


def build_page_title(section: str) -> str:
    return f"{section} | {APP_NAME}"
