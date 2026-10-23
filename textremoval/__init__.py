"""Single-pass scene text removal on top of RevealLayer.

See ``docs/METHOD.md`` for the method and ``scripts/remove_text.py`` for the command-line entry point.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

__all__ = ["REPO_ROOT"]
