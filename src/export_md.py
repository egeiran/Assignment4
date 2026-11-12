"""Export a Markdown file to HTML and PDF next to it, as the A1/A3 reports were:
python-markdown with the A1 stylesheet, then headless Chrome.

Usage: uv run python -B src/export_md.py predictions/predictions_v1.md
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

# Same stylesheet as Assignment1/assignment3/src/export_report.py.
STYLE = """<meta charset='utf-8'><style>
body{font-family:Helvetica,Arial,sans-serif;font-size:11pt;line-height:1.45;max-width:17cm;margin:2cm auto;color:#222}
h1,h2{border-bottom:1px solid #ccc;padding-bottom:.2em} h2{page-break-before:auto}
table{border-collapse:collapse;margin:1em 0;font-size:9.5pt} th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}
th{background:#f2f2f2} code{background:#f4f4f4;padding:1px 4px;font-size:9.5pt} pre{background:#f4f4f4;padding:8px;font-size:9pt}
img{max-width:100%} em{color:#555} hr{border:0;border-top:1px solid #ccc}
@media print{body{margin:0;max-width:none} h1,h2,h3{break-after:avoid} tr,img{break-inside:avoid}}
</style>"""
MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def main():
    source = Path(sys.argv[1]).resolve()
    html_path, pdf_path = source.with_suffix(".html"), source.with_suffix(".pdf")
    body = markdown.markdown(source.read_text(), extensions=["tables", "fenced_code"])
    html_path.write_text(STYLE + body)
    chrome = (os.environ.get("CHROME") or shutil.which("google-chrome")
              or shutil.which("chromium") or shutil.which("chromium-browser")
              or (MAC_CHROME if Path(MAC_CHROME).is_file() else None))
    if not chrome:
        raise SystemExit("HTML written; PDF requires Chrome/Chromium. Set CHROME to its executable path.")
    # Keep browser state temporary, and replace the PDF only after a successful export.
    with tempfile.TemporaryDirectory(prefix="a4-export-") as tmp:
        output = Path(tmp) / "report.pdf"
        subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={tmp}/profile", f"--print-to-pdf={output}",
                        html_path.as_uri()], check=True, capture_output=True)
        if not output.is_file():
            raise SystemExit("Chrome returned without writing a PDF; previous PDF preserved.")
        pdf_path.write_bytes(output.read_bytes())
    print("wrote", html_path, pdf_path)


if __name__ == "__main__":
    main()
