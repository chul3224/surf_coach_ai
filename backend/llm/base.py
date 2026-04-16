from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PoseData:
    """YOLO 분석 결과 → LLM 입력 구조"""
    action: str           # "popup" | "stance" | "paddling"
    scores: dict          # {"knee_angle": 145, "foot_gap_ratio": 1.8, ...}
    issues: list[str]     # ["시선이 아래를 향하고 있음", "무릎 각도 과도"]
    overall_score: float  # 0~100


@dataclass
class FeedbackResult:
    """LLM 피드백 출력 구조"""
    summary: str          # 한 줄 요약
    corrections: list[str]  # 교정 포인트 리스트
    encouragement: str    # 격려 메시지
    model_used: str       # 어떤 모델이 생성했는지


class BaseLLM(ABC):
    """모든 LLM 어댑터가 구현해야 할 공통 인터페이스"""

    @abstractmethod
    def get_feedback(self, pose_data: PoseData) -> FeedbackResult:
        raise NotImplementedError

    def _build_prompt(self, pose_data: PoseData) -> str:
        """공통 프롬프트 생성 — 모든 모델에서 동일하게 사용"""
        issues_text = "\n".join(f"- {issue}" for issue in pose_data.issues)
        scores_text = "\n".join(
            f"- {k}: {v}" for k, v in pose_data.scores.items()
        )
        return f"""당신은 경력 5년의 서핑 전문 강사입니다.
아래 자세 분석 결과를 바탕으로 초보 서퍼에게 구체적인 교정 피드백을 제공하세요.

[분석 동작]
{pose_data.action}

[측정값]
{scores_text}

[감지된 문제점]
{issues_text}

[종합 점수]
{pose_data.overall_score:.1f} / 100

응답 형식 (JSON):
{{
  "summary": "한 줄 요약 (20자 이내)",
  "corrections": ["교정 포인트 1", "교정 포인트 2", ...],
  "encouragement": "격려 메시지 (1문장)"
}}

서핑 전문 용어를 사용하되, 초보자도 이해할 수 있도록 설명하세요."""
