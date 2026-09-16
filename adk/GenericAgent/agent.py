from google.adk.agents.llm_agent import Agent


import os
from typing import List, Optional
import pandas as pd


def save_csv(data: List[dict], file_path: Optional[str] = None) -> dict:
    """
    Save structured data to a CSV file.

    Args:
        data: List of dictionaries
        file_path: Output file path

    Returns:
        JSON serializable response
    """

    try:
        if not data:
            return {
                "status": "error",
                "message": "No data provided"
            }

        if file_path is None:
            file_path = "../artifacts/results.csv"

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        df = pd.DataFrame(data)

        df.to_csv(file_path, index=False)

        return {
            "status": "success",
            "file_path": file_path,
            "rows_saved": len(df)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    
     
root_agent = Agent(
    model="gemini-2.5-flash",
    name="generic_research_agent",
    description="""
    Performs telecom industry research,
    market analysis,
    benchmarking,
    and competitor intelligence.
    """,
    instruction="""
You are a Telecom Industry Research Specialist.

Your objective is to gather external evidence and market intelligence relevant to the user's question.

Tasks:

1. Research telecom industry trends.
2. Identify customer experience best practices.
3. Identify competitor insights.
4. Identify industry benchmarks.
5. Extract evidence supporting findings.
6. Assign confidence scores.

Generate structured findings with:

- finding
- category
- source_summary
- business_relevance
- recommendation
- confidence_score

Categories:

- Industry Trend
- Competitor Insight
- Customer Experience Benchmark
- Market Intelligence

Business Relevance:

- High
- Medium
- Low

Output Requirements:

1. Generate structured JSON serializable data.
2. Include evidence summaries.
3. Include confidence scores.
4. Call save_csv tool.
5. Return save_csv response.


Example:

[
 {
   "finding":"Customers increasingly prefer self-service channels",
   "category":"Industry Trend",
   "source_summary":"Observed across multiple telecom reports",
   "business_relevance":"High",
   "recommendation":"Invest in digital self-service",
   "confidence_score":0.89
 }
]
""",
    tools=[save_csv]
)