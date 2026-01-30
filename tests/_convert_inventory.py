"""One-shot converter: parse tests_plan.md into tests/_inventory.json.

Run once then discard this script.

Usage:
    python tests/_convert_inventory.py
"""
import json
import re
import sys
from pathlib import Path

PLAN = Path(__file__).resolve().parent.parent / "tests_plan.md"
OUT = Path(__file__).resolve().parent / "_inventory.json"

# --- tag inference --------------------------------------------------------

_TAG_RULES = [
    # (regex on heading or text, tag)
    (re.compile(r"forwarding\s+propert", re.I), "forwarding"),
    (re.compile(r"table.driven", re.I), "forwarding"),
    (re.compile(r"delegates?\s+to", re.I), "forwarding"),
    (re.compile(r"\bupdate\b", re.I), "update"),
    (re.compile(r"\bbuild\b", re.I), "build"),
    (re.compile(r"\braises?\b", re.I), "error"),
    (re.compile(r"\bValueError\b"), "error"),
    (re.compile(r"\bKeyError\b"), "error"),
    (re.compile(r"\bTypeError\b"), "error"),
    (re.compile(r"\bImportError\b"), "error"),
    (re.compile(r"\bwhen\b.*\btrue\b|\bwhen\b.*\bfalse\b", re.I), "branch"),
    (re.compile(r"\bbranch\b", re.I), "branch"),
    (re.compile(r"\bno.op\b", re.I), "branch"),
    (re.compile(r"\bproperty\b", re.I), "property"),
    (re.compile(r"\b__init__\b"), "init"),
    (re.compile(r"\bconstruction\b", re.I), "init"),
]


def infer_tags(heading: str, text: str) -> list[str]:
    combined = f"{heading} {text}"
    tags = []
    for pattern, tag in _TAG_RULES:
        if pattern.search(combined) and tag not in tags:
            tags.append(tag)
    return tags


# --- markdown parsing -----------------------------------------------------

# Match file headings in multiple formats produced by merge:
#   ### `filename.py` [x]              (A1 / completed)
#   ## `path/filename.py`              (A2, A4, A5, A6)
#   ## N. `path/filename.py`           (A3, A7)
#   ## N. `src/cubie/path/__init__.py`  (A7 init files)
FILE_HDR = re.compile(
    r"^#{2,3}\s+(?:\d+\.\s+)?`([^`]+\.py)`\s*(?:\[.\])?\s*$"
)
# Match section headings (h3 or h4)
SECTION_HDR = re.compile(r"^#{3,4}\s+(.+)")
# Match table rows like: | 1 | Some text |
# Also handle forwarding tables: | 1 | prop | delegate |
TABLE_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(.+?)\s*\|"
)
# Table separator
TABLE_SEP = re.compile(r"^\|[-\s|]+\|$")


def parse_plan(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()

    inventory = {}  # file -> list of items
    current_file = None
    current_section = ""

    for line in lines:
        # File heading
        m = FILE_HDR.match(line)
        if m:
            current_file = m.group(1)
            # Normalise: strip src/cubie/ prefix if present
            current_file = re.sub(
                r"^src/cubie/", "", current_file
            )
            if current_file not in inventory:
                inventory[current_file] = []
            current_section = ""
            continue

        # Section heading
        m = SECTION_HDR.match(line)
        if m:
            current_section = m.group(1).strip()
            current_section = current_section.replace("\u2014", "--")
            continue

        # Table row
        if current_file is None:
            continue
        m = TABLE_ROW.match(line)
        if m and not TABLE_SEP.match(line):
            num = int(m.group(1))
            # Grab everything after the first number column
            # Re-split on | to handle 2-col vs 3-col tables
            cells = [c.strip() for c in line.split("|")[1:] if c.strip()]
            if len(cells) >= 3:
                # Forwarding table: | # | Property | Delegates to |
                text = f"{cells[1]} -> {cells[2]}"
            elif len(cells) >= 2:
                text = cells[1]
            else:
                continue

            # Normalise unicode to ASCII for Windows compat
            text = text.replace("\u2192", "->")
            text = text.replace("\u2014", "--")
            tags = infer_tags(current_section, text)
            item = {
                "file": current_file,
                "section": current_section,
                "num": num,
                "text": text,
                "tags": tags,
            }
            inventory[current_file].append(item)

    return inventory


def main():
    if not PLAN.exists():
        print(f"ERROR: {PLAN} not found", file=sys.stderr)
        sys.exit(1)

    inventory = parse_plan(PLAN)

    total = sum(len(v) for v in inventory.values())
    print(f"Parsed {len(inventory)} files, {total} items")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
