"""Word count of a report, excluding tables, figures and their captions.

  uv run python -B src/count_words.py report/evidence_brief.md

Not counted: table rows (lines starting with |), image lines and caption lines
(italic lines starting with *Table or *Figure). Everything else is counted.
"""
import re
import argparse
from pathlib import Path


def count_words(text, include_all=False):
    words = 0
    caption = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            caption = False
        if not include_all:
            if line.startswith(("*Table", "*Figure")):
                caption = True
            if caption or line.startswith(("|", "![")):
                continue
        # HTML print-layout tags are not prose.
        line = re.sub(r"<[^>]*>", "", line)
        words += len(re.findall(r"[^\W_][\w'’·.,%–-]*", line))
    return words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--all", action="store_true", help="include tables and captions (Part 5)")
    args = parser.parse_args()
    suffix = "words in the complete record" if args.all else "words outside tables and captions"
    print(f"{args.source}: {count_words(args.source.read_text(), args.all)} {suffix}")


if __name__ == "__main__":
    main()
