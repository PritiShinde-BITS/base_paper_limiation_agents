from google.adk.agents.llm_agent import Agent
from ContentLimitationsAgent.agent import root_agent as content_limitations_agent
from TechnicalAuditAgent.agent import root_agent as technical_audit_agent
from CitationAgent.agent import root_agent as citation_agent
from MasterAgent.tools import extract_main_paper_content
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    model="gemini-3.6-flash",
    name="master_orchestrator_agent",
    description="Trims the base paper to its main content, runs 3 specialist paper-limitation agents, and synthesizes the final numbered limitations list itself.",
    instruction="""
Always respond in English only, regardless of any language in the input.

You are the Master Orchestrator and Coordinator for scientific paper limitation analysis.

For every paper you receive:

0. The uploaded artifact IS the base paper — do not ask for a filename or look for "base"
   in the name. Call extract_main_paper_content with the filename of whatever artifact
   was uploaded. If multiple artifacts are uploaded, use the first one as the base paper.
   This returns paper_content trimmed to only the main content (through Conclusion,
   Limitations, and Ethical/Broader Impact Statement) -- References, Bibliography,
   Acknowledgments, and Appendix are already removed. If cutoff_found is false, use
   paper_content as returned; do not fetch the raw file yourself.

1. Call ALL of the following specialist agents, passing them ONLY the returned
   paper_content text (never the raw artifact/file):
   - content_limitations_agent (explicit, inferred, and peer-review limitations)
   - technical_audit_agent (theory, baselines/novelty, robustness, reproducibility/compute, presentation)
   
2. Collect all two outputs.

3. Synthesize the two outputs YOURSELF into ONE cohesive, non-redundant, NUMBERED list
   ("1. ...", "2. ...", each item separated by a blank line). Prioritize critical limitations
   that affect the paper's validity and reproducibility, resolve contradictions, and remove
   duplicates across the two agents' outputs.

4. Return only this final numbered list -- do not return the raw per-agent outputs, and do not
   call any other agent to do the merging for you.
""",
    tools=[extract_main_paper_content],
    sub_agents=[
        content_limitations_agent, technical_audit_agent,
    ],
)