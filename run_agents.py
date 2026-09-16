"""
run_agents.py
-------------
Runs the same 9-agent + Master-Coordinator limitation-generation pipeline
as LimAgents_Llama_9_Agents.ipynb (cells 4-15), but calls a hosted API
(Gemini free tier by default, or Claude) via llm_clients.generate()
instead of loading Llama-3-8B locally. Prompt text is copied verbatim
from the notebook; only the model call and chat-template tokens differ.

Usage:
    python build_dataset.py                     # once, to create the input CSV
    python run_agents.py --provider gemini       # generate limitations
    python run_agents.py --provider claude       # or use Claude instead

Output:
    df_neurips_limitations_multi_agent_9_agents.csv
"""

import argparse
import time
import pandas as pd

from llm_clients import paced_generate

INPUT_CSV = "df_neruips_21_22_final.csv"
OUTPUT_CSV = "df_neurips_limitations_multi_agent_9_agents.csv"
MAX_NEW_TOKENS = 512

# ---------------------------------------------------------------------------
# Agent prompts (verbatim from the notebook's cells 4-13; Llama-3 chat-
# template tokens are harmless to keep since llm_clients strips them before
# sending to Gemini/Claude, but they're kept here so the prompts stay
# identical to the original notebook for reproducibility/comparison).
# ---------------------------------------------------------------------------

def get_extractor_prompt(paper_content: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in scientific literature analysis. Your task is to carefully read the provided scientific article
and extract all explicitly stated limitations as mentioned by the authors. Focus on sections such as Discussion, Conclusion, or
Limitations. List each limitation verbatim, including direct quotes where possible, and provide a brief context (e.g., what aspect of
the study the limitation pertains to). Ensure accuracy and avoid inferring or adding limitations not explicitly stated. If no limitations
are mentioned, state this clearly.

Output Format:
Bullet points listing each limitation.
For each: Verbatim quote (if available), context (e.g., aspect of the study), and section reference.
If none: "No limitations explicitly stated in the article."

<|eot_id|><|start_header_id|>user<|end_header_id|>

Paper Content:
{paper_content}

Please extract and list the key limitations found in this paper. Be specific and provide clear reasoning for each limitation identified.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_analyzer_prompt(paper_content: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a critical scientific reviewer with expertise in research methodology and analysis. Your task is to analyze
the provided scientific article and identify potential limitations not explicitly stated by the authors. Focus on aspects such as study
design, sample size, data collection methods, statistical analysis, scope of findings, and underlying assumptions. For each inferred
limitation, provide a clear explanation of why it is a limitation and how it impacts the study's validity, reliability,
or generalizability. Ensure inferences are grounded in the article's content and avoid speculative assumptions.

Output Format:
Bullet points listing each inferred limitation.
For each: Description, explanation, and impact on the study.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Paper Content:
{paper_content}

Please provide a detailed analysis of the limitations in this research. Consider both obvious and subtle limitations that could affect the validity and applicability of the findings.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_reviewer_prompt(paper_content: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in open peer review with a focus on transparent and critical evaluation of scientific research.
Your task is to review the provided scientific article from the perspective of an external peer reviewer. Identify potential limitations
that might be raised in an open review process, considering common critiques such as reproducibility, transparency, generalizability,
or ethical considerations.

Output Format:
Bullet points listing each limitation.
For each: Description, why it's a concern, and alignment with peer review standards.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Paper Content:
{paper_content}

Please provide a critical review identifying the limitations and areas of concern in this research. Consider what a peer reviewer would highlight as weaknesses or areas needing improvement.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_citation_prompt(paper_content: str, cited_papers: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

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

<|eot_id|><|start_header_id|>user<|end_header_id|>

Current Paper Content:
{paper_content}

Cited/Citing Papers Information:
{cited_papers}

Please identify limitations of the CURRENT paper's methodology and ideas above -- not limitations of the cited or
citing papers themselves -- using the cited/citing papers only as supporting evidence. If no cited-paper
information is provided, state that clearly and infer limitations based on the current paper's own content alone.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_theory_assumptions_prompt(paper_content: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert auditor specializing in mathematical rigor, theoretical validity, and assumption scrutiny. Your tasks are to:

Trace each formal claim from theoretical foundation to objective to algorithm, ensuring logical consistency.
Verify definitions, assumptions, and properties for correctness and applicability.
Identify overly restrictive or unrealistic assumptions.
Flag proof gaps, undefined notation, missing terms, or invalid generalizations.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Paper Content:
{paper_content}

Provide limitations about THEORY & ASSUMPTIONS for the following content. Use only facts from the text.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_baselines_novelty_prompt(paper_content: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert auditor evaluating baseline comparisons, novelty, and comparative fairness. Your tasks are to:

Assess baseline coverage and fairness, verifying equivalence in parameters, data, tuning, random seeds, and metrics.
Ensure comparisons include state-of-the-art (SOTA) or standard methods relevant to the task.
Detect discrepancies or outdated results compared to cited prior work.
Evaluate novelty, identifying whether contributions are incremental, rebranded prior art, or truly novel, with evidence.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Paper Content:
{paper_content}

Provide limitations about BASELINES, NOVELTY & COMPARABILITY for the following content.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_robustness_failure_prompt(paper_content: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert auditor focused on robustness, failure modes, and ablation studies. Your tasks are to:

Identify missing tests for edge cases (e.g., noise, adversarial inputs, distribution shifts, extreme parameter limits, long sequences).
Verify sensitivity analyses for key hyperparameters and model components.
Ensure inclusion of simple heuristic baselines where relevant.
Confirm the presence of detailed failure case analyses with examples.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Paper Content:
{paper_content}

Provide limitations about ROBUSTNESS, FAILURE MODES & ABLATIONS for the following content.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_reproducibility_compute_prompt(paper_content: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert auditor verifying reproducibility, computational claims, and scalability. Your tasks are to:

Flag missing details on training/inference time, FLOPs, memory usage, hardware, batch sizes, random seeds,
confidence intervals, dataset sizes, or code/data availability.
Evaluate scalability, checking if results are limited to small instances or include a cost model for larger scales.
Flag unsubstantiated claims of "efficiency" or "faster" performance without quantitative evidence.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Paper Content:
{paper_content}

Provide limitations about REPRODUCIBILITY, COMPUTE & SCALABILITY for the following content.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_presentation_clarity_prompt(paper_content: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert auditor ensuring clarity, presentation integrity, and impact discussion. Your tasks are to:

Identify undefined terms, inconsistent notation, or mislabeled equations, figures, or tables.
Verify the inclusion of sufficient qualitative evidence.
Flag issues in captions or legends.
Ensure discussion of societal, ethical, or fairness impacts when relevant to the task or application.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Paper Content:
{paper_content}

Provide limitations about PRESENTATION, CLARITY & IMPACT for the following content.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_merger_prompt(extractor_output, analyzer_output, reviewer_output, citation_output,
                       theory_output, baselines_output, robustness_output,
                       reproducibility_output, presentation_output) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a **Master Coordinator**, an expert in scientific communication and synthesis. Your task is to integrate limitations provided by nine specialized agents:

1. Extractor (explicit limitations from the article)
2. Analyzer (inferred limitations from critical analysis)
3. Reviewer (limitations from an open review perspective)
4. Citation (limitations of the article's own methodology/ideas, grounded in its cited/citing papers)
5. Theory & Assumptions (mathematical correctness, assumptions, scope)
6. Baselines & Novelty (comparative evaluation, novelty assessment)
7. Robustness & Failure Modes (edge conditions, ablation studies)
8. Reproducibility & Compute (reproducibility claims, scalability)
9. Presentation & Clarity (clarity, integrity, impact discussion)

Goals:
1. Combine all limitations into a cohesive, non-redundant list.
2. Ensure each limitation is clearly stated, scientifically valid, and aligned with the article's content.
3. Prioritize critical limitations that affect the paper's validity and reproducibility.
4. Format the final list as a NUMBERED list ("1. ...", "2. ...", each item separated by a blank line),
   suitable for a scientific review or report.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Extractor Agent Analysis:
{extractor_output}

Analyzer Agent Analysis:
{analyzer_output}

Reviewer Agent Analysis:
{reviewer_output}

Citation Agent Analysis:
{citation_output}

Theory & Assumptions Agent Analysis:
{theory_output}

Baselines & Novelty Agent Analysis:
{baselines_output}

Robustness & Failure Modes Agent Analysis:
{robustness_output}

Reproducibility & Compute Agent Analysis:
{reproducibility_output}

Presentation & Clarity Agent Analysis:
{presentation_output}

Please merge these nine different perspectives on the paper's limitations into a comprehensive, well-organized,
NUMBERED list. Synthesize the insights, resolve any contradictions, and provide a unified view of the paper's limitations.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


AGENTS = [
    ("Extractor", get_extractor_prompt, "limitations_extractor"),
    ("Analyzer", get_analyzer_prompt, "limitations_analyzer"),
    ("Reviewer", get_reviewer_prompt, "limitations_reviewer"),
    ("Theory", get_theory_assumptions_prompt, "limitations_theory_assumptions"),
    ("Baselines", get_baselines_novelty_prompt, "limitations_baselines_novelty"),
    ("Robustness", get_robustness_failure_prompt, "limitations_robustness_failure"),
    ("Reproducibility", get_reproducibility_compute_prompt, "limitations_reproducibility_compute"),
    ("Presentation", get_presentation_clarity_prompt, "limitations_presentation_clarity"),
]


def run_agent(name, prompt_func, paper_content, provider):
    print(f"  Running {name} agent...")
    try:
        prompt = prompt_func(paper_content)
        out = paced_generate(prompt, max_new_tokens=MAX_NEW_TOKENS, provider=provider)
        print(f"  {name} agent completed ({len(out)} chars)")
        return out
    except Exception as e:
        print(f"  Error in {name} agent: {e}")
        return f"ERROR in {name} agent: {e}"


def build_combined_text(row) -> str:
    return (
        f"Abstract: {row.get('df_Abstract', '')}\n"
        f"Introduction: {row.get('df_Introduction', '')}\n"
        f"Related_Work: {row.get('df_Related_Work', '')}\n"
        f"Methodology: {row.get('df_Methodology', '')}\n"
        f"Dataset: {row.get('df_Dataset', '')}\n"
        f"Conclusion: {row.get('df_Conclusion', '')}\n"
        f"Experiment_and_Results: {row.get('df_Experiment_and_Results', '')}\n"
        f"LLM_extracted_future_work: {row.get('LLM_extracted_future_work', '')}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["gemini", "claude"], default="gemini")
    parser.add_argument("--input", default=INPUT_CSV)
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df["combined"] = df.apply(build_combined_text, axis=1)

    for _, col in [(a, c) for a, _, c in AGENTS]:
        df[col] = ""
    df["limitations_citation_only"] = ""
    df["limitations_merged_final"] = ""

    start = time.time()
    for i in range(len(df)):
        print(f"\n=== Processing row {i+1}/{len(df)} (provider={args.provider}) ===")
        row = df.iloc[i]
        paper_content = row["combined"]

        cited_in = row.get("relevance_8_cited_in", "")
        cited_by = row.get("relevance_8_cited_by", "")
        cited_papers = f"Papers cited by this article:\n{cited_in}\n\nPapers that cited this article:\n{cited_by}"
        if not str(cited_in).strip() and not str(cited_by).strip():
            cited_papers += "\n\n(No citation data was supplied for this run; see fetch_citations.py.)"

        outputs = {}
        for name, prompt_func, col in AGENTS:
            outputs[name] = run_agent(name, prompt_func, paper_content, args.provider)
            df.at[i, col] = outputs[name]

        citation_output = run_agent("Citation", lambda _: get_citation_prompt(paper_content, cited_papers), "", args.provider)
        df.at[i, "limitations_citation_only"] = citation_output

        print("  Running Master Coordinator agent...")
        try:
            merger_prompt = get_merger_prompt(
                outputs["Extractor"], outputs["Analyzer"], outputs["Reviewer"], citation_output,
                outputs["Theory"], outputs["Baselines"], outputs["Robustness"],
                outputs["Reproducibility"], outputs["Presentation"],
            )
            merged = paced_generate(merger_prompt, max_new_tokens=MAX_NEW_TOKENS, provider=args.provider)
            df.at[i, "limitations_merged_final"] = merged.strip()
            print("  Master Coordinator agent completed")
        except Exception as e:
            print(f"  Error in Master Coordinator agent: {e}")
            df.at[i, "limitations_merged_final"] = f"ERROR in Master Coordinator agent: {e}"

        df.to_csv(args.output, index=False)
        print(f"  Checkpoint saved to {args.output}")

    elapsed = time.time() - start
    print(f"\nDone. Total elapsed: {elapsed:.1f}s. Results saved to {args.output}")


if __name__ == "__main__":
    main()
