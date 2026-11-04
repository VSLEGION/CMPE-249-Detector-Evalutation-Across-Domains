"""Repository entry point; no installation or working-directory assumption."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cmpe249_eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
