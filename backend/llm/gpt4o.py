import json
import os
from openai import OpenAI
from .base import BaseLLM, PoseData, FeedbackResult


class GPT4oLLM(BaseLLM):
    """GPT-4o 어댑터 (개발/테스트용 baseline)"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"

    def get_feedback(self, pose_data: PoseData) -> FeedbackResult:
        prompt = self._build_prompt(pose_data)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        return FeedbackResult(
            summary=data["summary"],
            corrections=data["corrections"],
            encouragement=data["encouragement"],
            model_used=f"openai/{self.model}",
        )
