"""
VisionAI Centralized LLM Provider
──────────────────────────────────
Unified abstraction over OpenAI and Google Gemini APIs.
Supports text generation and vision (image + text) generation.
Automatic fallback: if the primary provider fails, try the other.
Provider can be switched at runtime from the UI without restarting.
"""

import os
import base64
import io
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


class LLMProvider:
    """
    Dual-provider LLM client supporting OpenAI and Gemini.

    Usage:
        llm = LLMProvider(provider="openai", config=config['models']['vlm'])
        text = llm.generate_text("Summarize this report...")
        desc = llm.generate_with_image(pil_image, "Describe this scene.")
    """

    PROVIDERS = ("openai", "gemini")

    def __init__(self, provider: str = "openai", config: dict = None):
        self.config = config or {}
        self.provider = provider.lower()
        if self.provider not in self.PROVIDERS:
            self.provider = "openai"

        
        self._openai_client = None
        self._gemini_model = None

        
        api_cfg = self.config.get("api", {})
        self.openai_model = api_cfg.get("openai", {}).get("model", "gpt-4.1")
        self.gemini_model_name = api_cfg.get("gemini", {}).get("model", "gemini-2.5-flash")

    
    
    

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def _get_gemini_model(self):
        if self._gemini_model is None:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key or api_key == "your_gemini_api_key_here":
                raise ValueError("GEMINI_API_KEY not found or is placeholder.")
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(self.gemini_model_name)
        return self._gemini_model

    
    
    

    def generate_text(self, prompt: str, system_prompt: str = None, max_tokens: int = 1024) -> str:
        """
        Generate text from a prompt string.
        Falls back to the alternate provider on failure.
        """
        providers_order = [self.provider] + [p for p in self.PROVIDERS if p != self.provider]

        for prov in providers_order:
            try:
                if prov == "openai":
                    return self._openai_text(prompt, system_prompt, max_tokens)
                else:
                    return self._gemini_text(prompt, system_prompt)
            except Exception as e:
                print(f"[LLMProvider] {prov} text generation failed: {e}")
                continue

        return "[LLMProvider] All providers failed. No response generated."

    def generate_with_image(self, image: Image.Image, prompt: str,
                            system_prompt: str = None, max_tokens: int = 1024) -> str:
        """
        Generate text from an image + text prompt (vision).
        Falls back to the alternate provider on failure.
        """
        providers_order = [self.provider] + [p for p in self.PROVIDERS if p != self.provider]

        for prov in providers_order:
            try:
                if prov == "openai":
                    return self._openai_vision(image, prompt, system_prompt, max_tokens)
                else:
                    return self._gemini_vision(image, prompt)
            except Exception as e:
                print(f"[LLMProvider] {prov} vision generation failed: {e}")
                continue

        return "[LLMProvider] All providers failed. No response generated."

    def switch_provider(self, new_provider: str):
        """Switch the primary provider at runtime."""
        new_provider = new_provider.lower()
        if new_provider in self.PROVIDERS:
            self.provider = new_provider

    
    
    

    def _openai_text(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        client = self._get_openai_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _openai_vision(self, image: Image.Image, prompt: str,
                       system_prompt: str, max_tokens: int) -> str:
        client = self._get_openai_client()

        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64_data}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}},
            ],
        })

        response = client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    
    
    

    def _gemini_text(self, prompt: str, system_prompt: str) -> str:
        model = self._get_gemini_model()
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = model.generate_content(full_prompt)
        return response.text.strip()

    def _gemini_vision(self, image: Image.Image, prompt: str) -> str:
        model = self._get_gemini_model()
        response = model.generate_content([prompt, image])
        return response.text.strip()
