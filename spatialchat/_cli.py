"""CLI entry point for SpatialChat.

Usage:
    uv run run-app          # starts the Streamlit app
    uv run run-app --port 8502
"""

import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    if not app_path.exists():
        print(f"Error: app.py not found at {app_path}", file=sys.stderr)
        sys.exit(1)

    # Forward any extra CLI args (e.g. --port 8502) to streamlit
    cmd = ["streamlit", "run", str(app_path)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
