"""Trims the uploaded base paper down to its main content (through Conclusion,
Limitations, and Ethical/Broader-Impact Statement), dropping References,
Bibliography, Acknowledgments, and Appendix to cut token usage before fan-out."""

import io
import re

from google.adk.tools.tool_context import ToolContext

# Matches a standalone section-heading line marking the end of "main content".
_CUTOFF_HEADING_RE = re.compile(
    r"(?im)^\s*(?:[0-9]+\.?\s*)?"
    r"(references|bibliography|acknowledg(?:e)?ments?|appendix[\s:]*[a-z]?)\s*$"
)


def _extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


async def extract_main_paper_content(artifact_name: str, tool_context: ToolContext) -> dict:
    """Loads the uploaded base paper artifact and returns only its main content.

    Keeps everything through Conclusion / Limitations / Ethical or Broader Impact
    Statement; drops References, Bibliography, Acknowledgments, and Appendix.

    Args:
        artifact_name: The filename of the uploaded base paper artifact.

    Returns:
        A dict with status, whether a cutoff heading was found, and the trimmed text.
    """
    artifact = await tool_context.load_artifact(artifact_name)
    if artifact is None or artifact.inline_data is None:
        return {"status": "error", "message": f"Artifact '{artifact_name}' not found."}

    mime_type = artifact.inline_data.mime_type or ""
    data = artifact.inline_data.data

    if "pdf" in mime_type:
        text = _extract_pdf_text(data)
    elif isinstance(data, (bytes, bytearray)):
        text = data.decode("utf-8", errors="ignore")
    else:
        text = str(data)

    match = _CUTOFF_HEADING_RE.search(text)
    trimmed = text[: match.start()].strip() if match else text.strip()

    return {
        "status": "success",
        "cutoff_found": bool(match),
        "original_chars": len(text),
        "trimmed_chars": len(trimmed),
        "paper_content": trimmed,
    }