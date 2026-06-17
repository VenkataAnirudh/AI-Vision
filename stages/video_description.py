"""
VisionAI — Video Description Stage
────────────────────────────────────
Uses the dual-provider LLM abstraction (OpenAI GPT-4.1 / Gemini 2.5 Flash)
for keyframe description and narrative synthesis.

Supports:
  - API mode: Vision LLM describes keyframes, text LLM synthesizes summary
  - Local mode: Moondream2 local inference (fallback for offline use)
"""

import torch
import numpy as np
from PIL import Image
from utils.llm_provider import LLMProvider


class VideoDescriber:
    def __init__(self, model_manager, config, llm_provider_name: str = "openai"):
        self.config = config['models']['vlm']
        self.manager = model_manager
        self.mode = self.config['mode']

        if self.mode == 'api':
            self.llm = LLMProvider(provider=llm_provider_name, config=self.config)
        else:
            self.model = self._load_local_vlm()

    def _load_local_vlm(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.config['local']['model_id']
        revision = self.config['local']['revision']

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        def loader():
            return AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, revision=revision, torch_dtype=torch.float16
            )

        return self.manager.load_torch_model('moondream', loader)

    def describe_keyframe(self, frame_rgb, prompt="Describe the actions, location, and people visible in this scene frame."):
        """
        Generates a natural language description of a single frame.
        Uses vision capability of the selected LLM provider.
        """
        pil_image = Image.fromarray(frame_rgb)

        if self.mode == 'api':
            system_prompt = (
                "You are an expert CCTV video analyst. Describe the scene precisely and concisely. "
                "Focus on: number of people, their actions, body language, environment, "
                "and any notable objects or events. Keep your response to 2-3 sentences."
            )
            return self.llm.generate_with_image(
                pil_image, prompt,
                system_prompt=system_prompt,
                max_tokens=256,
            )
        else:
            
            enc_image = self.model.encode_image(pil_image)
            return self.model.answer_question(enc_image, prompt, self.tokenizer)

    def synthesize_summary(self, frame_descriptions: list, events_summary: str, context_image=None) -> str:
        """
        Takes the per-frame textual descriptions and a summary of events,
        and prompts the LLM to write a cohesive narrative summary.
        and detected event logs using the selected LLM provider.
        """
        if not frame_descriptions:
            return "No keyframe descriptions available for narrative synthesis."

        prompt = f"""Based on these chronologically sequenced frame descriptions from a surveillance video:
{chr(10).join(frame_descriptions)}

Also, the following incidents were detected during automated analysis:
{events_summary if events_summary else "No significant incidents detected."}

Write a single concise paragraph (3-5 sentences) summarizing what happens in the video.
Reference timestamps or target identities where helpful. Keep the output formal and concise."""

        system_prompt = (
            "You are an expert security analyst writing a forensic video summary report. "
            "Be precise, reference timestamps, and maintain a professional tone."
        )

        if self.mode == 'api':
            if context_image is not None:
                img_to_encode = Image.fromarray(context_image) if isinstance(context_image, np.ndarray) else context_image
                return self.llm.generate_with_image(
                    img_to_encode,
                    prompt,
                    system_prompt=system_prompt,
                    max_tokens=512,
                )
            else:
                return self.llm.generate_text(
                    prompt,
                    system_prompt=system_prompt,
                    max_tokens=512,
                )
        else:
            
            img_to_encode = context_image if context_image is not None else Image.new('RGB', (10, 10))
            if isinstance(img_to_encode, np.ndarray):
                img_to_encode = Image.fromarray(img_to_encode)
            enc_img = self.model.encode_image(img_to_encode)
            return self.model.answer_question(enc_img, prompt, self.tokenizer)