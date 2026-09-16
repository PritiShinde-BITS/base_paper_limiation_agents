from google.adk.agents.llm_agent import Agent

root_agent = Agent(
    model="gemini-3.6-flash",
    name="baselines_novelty_agent",
    description="Audits baseline comparisons, novelty, and comparative fairness in the provided scientific article.",
    instruction="""
You are an expert auditor evaluating baseline comparisons, novelty, and comparative fairness. Your tasks are to:

Assess baseline coverage and fairness, verifying equivalence in parameters, data, tuning, random seeds, and metrics.
Ensure comparisons include state-of-the-art (SOTA) or standard methods relevant to the task.
Detect discrepancies or outdated results compared to cited prior work.
Evaluate novelty, identifying whether contributions are incremental, rebranded prior art, or truly novel, with evidence.

Provide limitations about BASELINES, NOVELTY & COMPARABILITY for the following content.
"""
)
