"""
API 라우터

POST /analyze      — 영상 업로드 → YOLO 자세 분석 → LLM 피드백 → 오버레이 이미지 반환
GET  /history      — 분석 기록 목록 조회
GET  /history/{id} — 단건 조회
"""

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..analysis import analyze, extract_keypoints_from_video, save_overlay_image
from ..analysis.yolo_extractor import extract_multi_keypoints_from_video, extract_popup_stage_frames
from ..analysis.popup_analyzer import analyze_popup_stages, analyze_popup_from_stage_frames
from ..db import get_db, AnalysisRecord
from ..llm import get_llm, PoseData

router = APIRouter()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
STATIC_DIR = Path("./static/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# POST /analyze
# ──────────────────────────────────────────────

@router.post("/analyze")
async def analyze_video(
    video: UploadFile = File(..., description="서핑 영상 파일"),
    action: str = Form(..., description="분석 동작: popup | stance | paddling"),
    db: AsyncSession = Depends(get_db),
):
    if action not in ("popup", "stance", "paddling"):
        raise HTTPException(
            status_code=400,
            detail="action 은 popup / stance / paddling 중 하나여야 합니다."
        )

    # 1) 영상 저장
    ext = Path(video.filename).suffix or ".mp4"
    file_id = uuid.uuid4().hex
    video_path = UPLOAD_DIR / f"{file_id}{ext}"
    with open(video_path, "wb") as f:
        f.write(await video.read())

    # 2) YOLO 키포인트 추출 + 자세 분석
    try:
        if action == "popup":
            # 팝업: body metrics 기반 3단계 자동 감지 → 단계별 분석
            stage_frames = extract_popup_stage_frames(str(video_path))
            stage_kps = {s: kps for s, (kps, _) in stage_frames.items()}
            popup_result = analyze_popup_from_stage_frames(stage_kps)

            # 오버레이: Stage 2(쭈그린 자세) 프레임 사용 — 팝업의 핵심 순간
            keypoints_raw, frame = stage_frames[2]

            # AnalysisResult 호환 구조로 변환
            analysis_action = popup_result.action
            analysis_scores = {
                **popup_result.scores,
                **{f"1단계_{k}": v for k, v in popup_result.stages[0].scores.items()},
                **{f"2단계_{k}": v for k, v in popup_result.stages[1].scores.items()},
                **{f"3단계_{k}": v for k, v in popup_result.stages[2].scores.items()},
            }
            analysis_issues = popup_result.issues
            analysis_overall = popup_result.overall_score
            stages_detail = [
                {
                    "stage": s.stage,
                    "name": s.stage_name,
                    "score": s.overall_score,
                    "issues": s.issues,
                }
                for s in popup_result.stages
            ]
        else:
            # stance / paddling: 단일 프레임 분석
            keypoints_raw, frame = extract_keypoints_from_video(str(video_path))
            result = analyze(action, keypoints_raw)
            analysis_action = result.action
            analysis_scores = result.scores
            analysis_issues = result.issues
            analysis_overall = result.overall_score
            stages_detail = None

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 3) 오버레이 이미지 생성
    overlay_filename = f"{file_id}_overlay.jpg"
    overlay_path = str(STATIC_DIR / overlay_filename)
    save_overlay_image(
        frame=frame,
        keypoints=keypoints_raw,
        action=action,
        scores=analysis_scores,
        overall_score=analysis_overall,
        save_path=overlay_path,
    )
    overlay_url = f"/static/results/{overlay_filename}"

    # 4) LLM 피드백
    pose_data = PoseData(
        action=analysis_action,
        scores=analysis_scores,
        issues=analysis_issues,
        overall_score=analysis_overall,
    )
    llm = get_llm()
    feedback = llm.get_feedback(pose_data)

    # 5) DB 저장
    record = AnalysisRecord(
        video_filename=str(video_path.name),
        action=action,
        overall_score=analysis_overall,
        scores=analysis_scores,
        issues=analysis_issues,
        overlay_image_url=overlay_url,
        feedback_summary=feedback.summary,
        feedback_corrections=feedback.corrections,
        feedback_encouragement=feedback.encouragement,
        model_used=feedback.model_used,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    response = {
        "id": record.id,
        "action": action,
        "overall_score": analysis_overall,
        "scores": analysis_scores,
        "issues": analysis_issues,
        "overlay_image_url": overlay_url,
        "feedback": {
            "summary": feedback.summary,
            "corrections": feedback.corrections,
            "encouragement": feedback.encouragement,
            "model_used": feedback.model_used,
        },
    }
    if stages_detail:
        response["popup_stages"] = stages_detail

    return response


# ──────────────────────────────────────────────
# GET /history
# ──────────────────────────────────────────────

@router.get("/history")
async def get_history(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(AnalysisRecord)
        .order_by(AnalysisRecord.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    records = result.scalars().all()

    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "action": r.action,
            "overall_score": r.overall_score,
            "overlay_image_url": r.overlay_image_url,
            "feedback_summary": r.feedback_summary,
        }
        for r in records
    ]


@router.get("/history/{record_id}")
async def get_record(record_id: int, db: AsyncSession = Depends(get_db)):
    record = await db.get(AnalysisRecord, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="기록을 찾을 수 없습니다.")

    return {
        "id": record.id,
        "created_at": record.created_at.isoformat(),
        "action": record.action,
        "overall_score": record.overall_score,
        "scores": record.scores,
        "issues": record.issues,
        "overlay_image_url": record.overlay_image_url,
        "feedback": {
            "summary": record.feedback_summary,
            "corrections": record.feedback_corrections,
            "encouragement": record.feedback_encouragement,
            "model_used": record.model_used,
        },
    }
