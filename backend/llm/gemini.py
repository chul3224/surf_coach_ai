import json
import os
import google.generativeai as genai
from .base import BaseLLM, PoseData, FeedbackResult


class GeminiLLM(BaseLLM):
    """Gemini 어댑터 (중간 성능 검증용)"""

    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")

    def get_feedback(self, pose_data: PoseData) -> FeedbackResult:
        prompt = self._build_prompt(pose_data)

        response = self.model.generate_content(prompt)
        raw = response.text

        # Gemini는 ```json ... ``` 블록으로 감싸는 경우가 있어 파싱 처리
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        data = json.loads(raw)

        return FeedbackResult(
            summary=data["summary"],
            corrections=data["corrections"],
            encouragement=data["encouragement"],
            model_used="google/gemini-1.5-flash",
        )
