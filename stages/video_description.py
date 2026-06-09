import os
import torch
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

class VideoDescriber:
    def __init__(self, model_manager, config):
        load_dotenv()
        self.config = config['models']['vlm']
        self.manager = model_manager
        self.mode = self.config['mode']

        if self.mode == 'api':
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            self.api_model = genai.GenerativeModel(self.config['api']['model'])
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

    def describe_keyframe(self, frame_rgb, prompt="Describe what is happening in this scene."):
        """
        Generates a natural language description of a single frame.
        """
        pil_image = Image.fromarray(frame_rgb)
        
        if self.mode == 'api':
            try:
                response = self.api_model.generate_content([prompt, pil_image])
                return response.text.strip()
            except Exception as e:
                print(f"[VideoDescriber] Gemini API frame description error: {e}")
                return "Frame description unavailable (API error)."
        else:
            # Moondream2 local inference
            enc_image = self.model.encode_image(pil_image)
            return self.model.answer_question(enc_image, prompt, self.tokenizer)

    def synthesize_summary(self, frame_descriptions, events_summary):
        """
        Synthesizes a master summary paragraph from frame-by-frame narrative
        and detected event logs.
        """
        prompt = f"""You are an expert video analyst. Based on these chronologically sequenced frame descriptions:
{chr(10).join(frame_descriptions)}

Also, the following incidents were detected during analysis:
{events_summary}

Write a single concise paragraph (3-5 sentences) summarizing what happens in the video.
Reference timestamps or target identities where helpful. Keep the output formal and concise."""

        if self.mode == 'api':
            try:
                return self.api_model.generate_content(prompt).text.strip()
            except Exception as e:
                print(f"[VideoDescriber] Gemini API synthesis error: {e}")
                return "Master summary generation failed due to API error."
        else:
            # Local fallback for text-only synthesis (Moondream is image-first, using dummy image)
            dummy_img = Image.new('RGB', (10, 10))
            enc_dummy = self.model.encode_image(dummy_img)
            return self.model.answer_question(enc_dummy, prompt, self.tokenizer)