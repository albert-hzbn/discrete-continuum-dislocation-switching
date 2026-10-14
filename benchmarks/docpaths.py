"""Locations of the generated result documents under docs/.

Report scripts compose their Markdown with repository-root-relative links
(for example ``results/<run>/report.json`` or ``PERFORMANCE_PROTOCOL.md``).
``write_doc`` places the document at its folder from docs/moves.json and
rewrites those links so they resolve from that folder.
"""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def _moves():
    return json.loads((ROOT / "docs" / "moves.json").read_text(encoding="utf-8"))["moved"]


def doc_path(name):
    """Repository path of a generated document, given its file name."""
    return ROOT / _moves().get(name, name)


def relink(text, name):
    """Rewrite root-relative links in text for a document stored at doc_path(name)."""
    moves = _moves()
    folder = os.path.dirname(moves.get(name, name))

    def fix(match):
        target = match.group(2)
        if re.match(r"^[a-z]+:", target) or target.startswith(("#", "/", "..")):
            return match.group(0)
        path, _, anchor = target.partition("#")
        if not path:
            return match.group(0)
        rel = os.path.relpath(moves.get(path, path), folder or ".").replace("\\", "/")
        return match.group(1) + rel + ("#" + anchor if anchor else "") + match.group(3)

    return _LINK.sub(fix, text)


def write_doc(name, text):
    """Write a generated result document to its docs/ location."""
    path = doc_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(relink(text, name), encoding="utf-8")
    return path
