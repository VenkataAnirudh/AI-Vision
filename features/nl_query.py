"""
VisionAI — Natural Language Report Query Agent
────────────────────────────────────────────────
Loads a structured JSON analysis report and answers
natural language questions using the dual LLM provider
(OpenAI GPT-4.1 / Gemini 2.5 Flash).
"""

import os
import json
from utils.llm_provider import LLMProvider


class ReportQueryAgent:
    def __init__(self, config=None, llm_provider_name: str = "openai"):
        vlm_config = {}
        if config:
            vlm_config = config.get('models', {}).get('vlm', {})

        self.llm = LLMProvider(provider=llm_provider_name, config=vlm_config)

    def query_report(self, report_json_path: str, question: str) -> str:
        """
        Loads the structured JSON analysis report of a video and queries
        the LLM for an answer to a natural language question.
        """
        if not os.path.exists(report_json_path):
            return "Analysis report file not found."

        with open(report_json_path, 'r') as f:
            report_data = json.load(f)

        # Truncate large timeline data to stay within context limits
        if 'threat_timeline' in report_data and len(report_data['threat_timeline']) > 60:
            # Keep every Nth point to summarize the timeline
            step = len(report_data['threat_timeline']) // 60
            report_data['threat_timeline'] = report_data['threat_timeline'][::step]

        system_prompt = (
            "You are an expert security and video forensics analyst. You have access to a "
            "structured JSON analysis report of a surveillance video. Answer the user's question "
            "precisely, referencing timestamps, confidence levels, identified persons, and detected "
            "events where relevant. If the data does not support a conclusion, state that the "
            "information is not present in the analysis. Do not invent details."
        )

        prompt = f"""JSON Analysis Report:
{json.dumps(report_data, indent=2)}

Question: {question}

Answer factually based on the report data above."""

        return self.llm.generate_text(
            prompt,
            system_prompt=system_prompt,
            max_tokens=1024,
        )