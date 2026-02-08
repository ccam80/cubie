"""Query the test inventory for specific files or tags.

Usage:
    python tests/query_inventory.py <source_file>
    python tests/query_inventory.py --tag forwarding
    python tests/query_inventory.py --tag update
    python tests/query_inventory.py --tag error
    python tests/query_inventory.py --method update
    python tests/query_inventory.py --list-files
    python tests/query_inventory.py --list-tags

Tags are auto-inferred: forwarding, update, build, error,
branch, property, init.

The --method flag searches section headings for a method name.
"""
import json
import re
import sys
from pathlib import Path

INVENTORY = Path(__file__).resolve().parent / "_inventory.json"


def load() -> dict:
    if not INVENTORY.exists():
        print(
            f"ERROR: {INVENTORY} not found.\n"
            "Run: python tests/_convert_inventory.py",
            file=sys.stderr,
        )
        sys.exit(1)
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def query_file(data: dict, filename: str) -> list[dict]:
    """Return items for a source file (fuzzy match on suffix)."""
    # Exact match first
    if filename in data:
        return data[filename]
    # Suffix match
    for key, items in data.items():
        if key.endswith(filename) or key.endswith(f"/{filename}"):
            return items
    return []


def query_tag(data: dict, tag: str) -> list[dict]:
    """Return all items across all files matching a tag."""
    results = []
    for items in data.values():
        for item in items:
            if tag in item.get("tags", []):
                results.append(item)
    return results


def query_method(data: dict, method: str) -> list[dict]:
    """Return all items whose section heading contains method name."""
    pattern = re.compile(re.escape(method), re.I)
    results = []
    for items in data.values():
        for item in items:
            if pattern.search(item.get("section", "")):
                results.append(item)
    return results


def format_items(items: list[dict]) -> str:
    if not items:
        return "No matching items."
    lines = []
    current_file = None
    current_section = None
    for item in items:
        if item["file"] != current_file:
            current_file = item["file"]
            lines.append(f"\n### {current_file}")
            current_section = None
        if item["section"] != current_section:
            current_section = item["section"]
            lines.append(f"  {current_section}")
        tags = f"  [{', '.join(item['tags'])}]" if item["tags"] else ""
        lines.append(f"    {item['num']:3d}. {item['text']}{tags}")
    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    data = load()

    if args[0] == "--list-files":
        for f in sorted(data.keys()):
            print(f"  {f} ({len(data[f])} items)")
        return

    if args[0] == "--list-tags":
        tags = set()
        for items in data.values():
            for item in items:
                tags.update(item.get("tags", []))
        for t in sorted(tags):
            count = sum(
                1 for items in data.values()
                for item in items if t in item.get("tags", [])
            )
            print(f"  {t} ({count} items)")
        return

    if args[0] == "--tag" and len(args) >= 2:
        items = query_tag(data, args[1])
        print(format_items(items))
        return

    if args[0] == "--method" and len(args) >= 2:
        items = query_method(data, args[1])
        print(format_items(items))
        return

    # Default: treat as filename
    items = query_file(data, args[0])
    print(format_items(items))


if __name__ == "__main__":
    main()
