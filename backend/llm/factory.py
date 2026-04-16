import os
from .base import BaseLLM


def get_llm() -> BaseLLM:
    """환경변수 LLM_PROVIDER 를 읽어 알맞은 어댑터 반환.

    개발: LLM_PROVIDER=claude  (또는 gpt4o / gemini)
    배포: LLM_PROVIDER=gemma4  (기본값)
    """
    provider = os.getenv("LLM_PROVIDER", "gemma4").lower()

    if provider == "claude":
        from .claude import ClaudeLLM
        return ClaudeLLM()

    if provider == "gpt4o":
        from .gpt4o import GPT4oLLM
        return GPT4oLLM()

    if provider == "gemini":
        from .gemini import GeminiLLM
        return GeminiLLM()

    if provider == "gemma4":
        from .gemma4 import Gemma4LLM
        return Gemma4LLM()

    raise ValueError(
        f"지원하지 않는 LLM_PROVIDER: '{provider}'. "
        "사용 가능한 값: claude / gpt4o / gemini / gemma4"
    )
