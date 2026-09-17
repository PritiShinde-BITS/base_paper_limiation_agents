# tools.py
"""Reads every cited/citing paper PDF from the repo's citation folder (one file
at a time) and returns only its Introduction and Limitations sections, dropping
Abstract, Methodology, Results, Conclusion, References, Acknowledgments, and
Appendix to cut token usage.

The base paper is NOT handled here -- it's passed directly to the agent by the
master agent in the user message.
"""

import io
import os
import re

# Citation folder, relative to this file's location in the repo:
#   CitationAgent/tools.py -> CitationAgent/citations/
# Override with the CITATION_FOLDER_PATH env var if the folder lives elsewhere.
_CITATION_DIR = os.environ.get(
    "CITATION_FOLDER_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "citations"),
)

_KEEP_SECTIONS = {"introduction", "limitations", "limitation"}

# Matches a standalone section-heading line (optionally numbered).
_HEADING_RE = re.compile(
    r"(?im)^\s*(?:[0-9]+\.?\s*)?"
    r"(abstract|introduction|related work|background|method(?:ology)?|approach|"
    r"experiments?|results?|evaluation|discussion|conclusion|limitations?|"
    r"ethic(?:al|s) statement|broader impact|acknowledg(?:e)?ments?|references|"
    r"bibliography|appendix[\s:]*[a-z]?)\s*$"
)


def _extract_pdf_text(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_key_sections(text: str) -> str:
    """Keeps only the Introduction and Limitations sections of `text`."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return text.strip()

    kept = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip().lower()
        if heading not in _KEEP_SECTIONS:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        kept.append(text[m.start():end].strip())

    return "\n\n".join(kept).strip()


def extract_citation_key_sections() -> dict:
    """Walks the repo's citation folder and, for each PDF found, extracts only
    its Introduction and Limitations sections.

    Files are read one at a time from `_CITATION_DIR` (a folder relative to
    this repo, overridable via the CITATION_FOLDER_PATH env var). The base
    paper is intentionally excluded here -- it's provided to the agent
    directly by the master agent -- but any filename containing "base" is
    also skipped defensively in case it ends up in the folder.

    Returns:
        A dict mapping each citation filename to its trimmed sections.
    """
    if not os.path.isdir(_CITATION_DIR):
        return {
            "status": "error",
            "citation_papers": {},
            "message": f"Citation folder not found: {_CITATION_DIR}",
        }

    citation_files = sorted(
        f for f in os.listdir(_CITATION_DIR)
        if f.lower().endswith(".pdf") and "base" not in f.lower()
    )

    if not citation_files:
        return {
            "status": "success",
            "citation_papers": {},
            "message": f"No citation PDFs found in {_CITATION_DIR}.",
        }

    citation_papers = {}
    for filename in citation_files:
        path = os.path.join(_CITATION_DIR, filename)
        try:
            text = _extract_pdf_text(path)
        except Exception as exc:
            citation_papers[filename] = f"[error reading file: {exc}]"
            continue
        citation_papers[filename] = _extract_key_sections(text)

    return {"status": "success", "citation_papers": citation_papers}