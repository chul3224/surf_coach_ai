import json
import os
from .base import BaseLLM, PoseData, FeedbackResult


class Gemma4LLM(BaseLLM):
    """Gemma4 로컬 어댑터 (최종 배포 목표 모델)

    로컬 실행 방식:
    - Ollama 사용 시: ollama pull gemma3 후 REST API 호출
    - 직접 로드 시: HuggingFace transformers 사용
    현재는 Ollama REST API 방식으로 구현.
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL_NAME = "gemma3"  # ollama pull gemma3

    def __init__(self):
        import httpx
        self.client = httpx.Client(timeout=120.0)

    def get_feedback(self, pose_data: PoseData) -> FeedbackResult:
        prompt = self._build_prompt(pose_data)

        payload = {
            "model": self.MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }

        response = self.client.post(self.OLLAMA_URL, json=payload)
        response.raise_for_status()

        raw = response.json()["response"]
        data = json.loads(raw)

        return FeedbackResult(
            summary=data["summary"],
            corrections=data["corrections"],
            encouragement=data["encouragement"],
            model_used=f"local/{self.MODEL_NAME}",
        )
