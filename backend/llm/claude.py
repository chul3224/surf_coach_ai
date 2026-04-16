import json
import os
import anthropic
from .base import BaseLLM, PoseData, FeedbackResult


class ClaudeLLM(BaseLLM):
    """Claude API 어댑터 (개발/테스트용 baseline)"""

    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = "claude-sonnet-4-6"

    def get_feedback(self, pose_data: PoseData) -> FeedbackResult:
        prompt = self._build_prompt(pose_data)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text
        data = json.loads(raw)

        return FeedbackResult(
            summary=data["summary"],
            corrections=data["corrections"],
            encouragement=data["encouragement"],
            model_used=f"claude/{self.model}",
        )
