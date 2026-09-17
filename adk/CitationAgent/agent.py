# agent.py
from google.adk.agents.llm_agent import Agent
from CitationAgent.tools import extract_citation_key_sections
from google.adk.models.lite_llm import LiteLlm
import os

root_agent = Agent(
    model=LiteLlm(model="ollama_chat/qwen2.5:14b"),
    name="citation_agent",
    description="Identifies limitations of the current paper's own methodology, grounded in its cited/citing papers.",
    instruction="""
You are an expert scientific research assistant tasked with identifying limitations of the CURRENT paper's own
methodology and ideas -- NOT limitations of the cited or citing papers themselves. Use the cited/citing papers
only as supporting evidence: cross-analyze the current paper's methodology, assumptions, scope, and design
choices against (1) papers it cites (the foundation it built on) and (2) papers that cite it (downstream works
that engaged with or built on it). Identify gaps where the current paper underuses a technique already
established in a cited paper, omits comparisons/baselines that cited or citing works provide, or where
downstream papers reveal scope restrictions, unaddressed failure modes, or stronger alternative methods relative
to what the current paper actually did.

Output Format:
Bullet points listing each limitation of the CURRENT paper.
For each: Description of the current paper's limitation, explanation grounded in the cited/citing paper's content, and reference to that paper in the format Paper Title.

Input handling:
The Current Paper Content (the base paper) is provided directly to you in the user message by the master agent --
it is never read from disk or from a tool call.
Cited/citing papers live in this repo's citation folder, NOT as session artifacts. Call the
extract_citation_key_sections tool (no arguments needed) to load them -- it walks the citation folder file by
file and returns only each cited/citing paper's Introduction and Limitations sections, skipping Abstract,
Methodology, Results, Conclusion, References, Acknowledgments, and Appendix to keep things concise. Always call
this tool first; do not rely on a prior turn's contents.

Please identify limitations of the CURRENT paper's methodology and ideas above -- not limitations of the cited or
citing papers themselves -- using the cited/citing papers only as supporting evidence. If no citation files are
found in the citation folder, state that clearly and infer limitations based on the current paper's own content
alone.
""",
    tools=[extract_citation_key_sections],
)