"""Word count of a report, excluding tables, figures and their captions.

  uv run python -B src/count_words.py report/evidence_brief.md

Not counted: table rows (lines starting with |), image lines and caption lines
(italic lines starting with *Table or *Figure). Everything else is counted.
"""
import re
import sys
from pathlib import Path


def main():
    words = 0
    for line in Path(sys.argv[1]).read_text().splitlines():
        line = line.strip()
        if line.startswith(("|", "![", "*Table", "*Figure")):
            continue
        words += len(re.findall(r"[^\W_][\w'’·.,%–-]*", line))
    print(f"{sys.argv[1]}: {words} words outside tables and captions")


if __name__ == "__main__":
    main()
