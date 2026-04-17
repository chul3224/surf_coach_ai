from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    """영상 분석 결과 저장 테이블"""
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 영상 정보
    video_filename = Column(String(255), nullable=False)
    action = Column(String(50), nullable=False)  # takeoff / stance / paddling

    # 분석 결과
    overall_score = Column(Float, nullable=False)
    scores = Column(JSON, nullable=False)       # 세부 측정값
    issues = Column(JSON, nullable=False)       # 문제점 리스트

    # 시각화 이미지
    overlay_image_url = Column(String(500))

    # LLM 피드백
    feedback_summary = Column(String(200))
    feedback_corrections = Column(JSON)
    feedback_encouragement = Column(String(300))
    model_used = Column(String(100))
