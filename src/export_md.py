"""Export a Markdown file to HTML and PDF next to it, as the A1/A3 reports were:
python-markdown with the A1 stylesheet, then headless Chrome.

Usage: uv run python -B src/export_md.py predictions/predictions_v1.md
"""
import subprocess
import sys
from pathlib import Path

import markdown

# Same stylesheet as Assignment1/assignment3/src/export_report.py.
STYLE = """<meta charset='utf-8'><style>
body{font-family:Helvetica,Arial,sans-serif;font-size:11pt;line-height:1.45;max-width:17cm;margin:2cm auto;color:#222}
h1,h2{border-bottom:1px solid #ccc;padding-bottom:.2em} h2{page-break-before:auto}
table{border-collapse:collapse;margin:1em 0;font-size:9.5pt} th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}
th{background:#f2f2f2} code{background:#f4f4f4;padding:1px 4px;font-size:9.5pt} pre{background:#f4f4f4;padding:8px;font-size:9pt}
img{max-width:100%} em{color:#555} hr{border:0;border-top:1px solid #ccc}
</style>"""
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def main():
    source = Path(sys.argv[1]).resolve()
    html_path, pdf_path = source.with_suffix(".html"), source.with_suffix(".pdf")
    body = markdown.markdown(source.read_text(), extensions=["tables", "fenced_code"])
    html_path.write_text(STYLE + body)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   check=True, capture_output=True)
    print("wrote", html_path, pdf_path)


if __name__ == "__main__":
    main()
