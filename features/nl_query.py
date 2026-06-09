import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

class ReportQueryAgent:
    def __init__(self, config=None):
        load_dotenv()
        
        # Load API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            
        # Get model configuration or fallback to defaults
        self.config = config.get('models', {}).get('vlm', {}) if config else {}
        self.model_name = self.config.get('api', {}).get('model', 'gemini-1.5-flash')
        
        try:
            self.client_initialized = bool(api_key)
            if self.client_initialized:
                self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            print(f"[ReportQueryAgent] Error configuring Gemini API: {e}")
            self.client_initialized = False

    def query_report(self, report_json_path: str, question: str) -> str:
        """
        Loads the structured JSON analysis report of a video and queries
        the Gemini API for an answer to a natural language question.
        """
        if not os.path.exists(report_json_path):
            return "Analysis report file not found."
            
        with open(report_json_path, 'r') as f:
            report_data = json.load(f)
            
        if not self.client_initialized:
            return ("Gemini API key is not configured. Please set the GEMINI_API_KEY "
                    "environment variable or in the .env file to enable report querying.")

        prompt = f"""You are an expert security and video analyst. You have access to a structured JSON analysis report of a video. 
Answer the user's question precisely, referencing timestamps, confidence levels, and identified persons where relevant.

JSON Report:
{json.dumps(report_data, indent=2)}

Question: {question}

Answer factually. If the data does not support a conclusion, state that the information is not present. Do not invent details."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error querying Gemini API: {str(e)}"