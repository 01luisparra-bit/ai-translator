import sys
from pathlib import Path

# Root of the repo (ai-translator/)
ROOT = Path(__file__).resolve().parent.parent
# Points to: src/python
SRC_PYTHON = ROOT / "src" / "python"
# Points to: src/python/ai_translator
AI_TRANSLATOR = SRC_PYTHON / "ai_translator"

# Add both paths to sys.path if not already there
for p in (SRC_PYTHON, AI_TRANSLATOR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
