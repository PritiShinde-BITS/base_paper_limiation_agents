"""Trims cited/citing paper artifacts down to only Abstract, Introduction, and
Limitations sections, dropping Methodology, Results, Conclusion, References,
Acknowledgments, and Appendix to cut token usage."""

import io
import re

from google.adk.tools.tool_context import ToolContext

_KEEP_SECTIONS = {"abstract", "introduction", "limitations", "limitation"}

# Matches a standalone section-heading line (optionally numbered).
_HEADING_RE = re.compile(
    r"(?im)^\s*(?:[0-9]+\.?\s*)?"
    r"(abstract|introduction|related work|background|method(?:ology)?|approach|"
    r"experiments?|results?|evaluation|discussion|conclusion|limitations?|"
    r"ethic(?:al|s) statement|broader impact|acknowledg(?:e)?ments?|references|"
    r"bibliography|appendix[\s:]*[a-z]?)\s*$"
)


def _extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_key_sections(text: str) -> str:
    """Keeps only the Abstract, Introduction, and Limitations sections of `text`."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return text.strip()

    kept = []

    # Text before the first heading is usually the (unlabeled) abstract.
    first_heading = matches[0].group(1).strip().lower()
    if first_heading != "abstract" and text[: matches[0].start()].strip():
        kept.append(text[: matches[0].start()].strip())

    for i, m in enumerate(matches):
        heading = m.group(1).strip().lower()
        if heading not in _KEEP_SECTIONS:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        kept.append(text[m.start():end].strip())

    return "\n\n".join(kept).strip()


async def extract_citation_key_sections(tool_context: ToolContext) -> dict:
    """Loads every uploaded citation artifact (cited/citing papers) and returns
    only their Abstract, Introduction, and Limitations sections.

    Skips the base paper artifact (filename containing "base") and drops
    everything else (Methodology, Results, Conclusion, References, Appendix, etc.)
    from the remaining artifacts.

    Returns:
        A dict mapping each citation artifact filename to its trimmed sections.
    """
    artifact_names = await tool_context.list_artifacts()
    citation_names = [n for n in artifact_names if "base" not in n.lower()]

    if not citation_names:
        return {
            "status": "success",
            "citation_papers": {},
            "message": "No cited/citing paper artifacts found.",
        }

    citation_papers = {}
    for name in citation_names:
        artifact = await tool_context.load_artifact(name)
        if artifact is None or artifact.inline_data is None:
            continue

        mime_type = artifact.inline_data.mime_type or ""
        data = artifact.inline_data.data

        if "pdf" in mime_type:
            text = _extract_pdf_text(data)
        elif isinstance(data, (bytes, bytearray)):
            text = data.decode("utf-8", errors="ignore")
        else:
            text = str(data)

        citation_papers[name] = _extract_key_sections(text)

    return {"status": "success", "citation_papers": citation_papers}