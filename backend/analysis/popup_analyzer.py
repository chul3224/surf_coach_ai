"""
팝업(Pop-up) 3단계 세부 분석 모듈

강사 도메인 지식 기반 체크포인트:

1단계 — 푸쉬(Push)
  - 양손이 갈비뼈(어깨와 엉덩이 사이) 옆에 위치
  - 팔만 밀어서 상체를 들어올림
  - 체중이 뒤로 쏠리지 않도록 손 위치 확인

2단계 — 발 끌어오기(Pull & Squat)
  - 무릎이 몸 쪽으로 당겨지며 쭈그리는 자세
  - 시선이 아래(바닥)를 보면 중심 잃고 빠짐 → 핵심 체크

3단계 — 일어서기(Stand Up)
  - 쭈그린 자세에서 일어나는 동안 시선 처리
  - 바닥을 보면 중심 잃음 → 핵심 체크
  - 무릎 각도가 90~120도로 안정화되는지
"""

import math
import numpy as np
from dataclasses import dataclass, field
from .pose_analyzer import KeyPoint, AnalysisResult, _angle, _distance, _visible


@dataclass
class PopupStageResult:
    """팝업 3단계 각각의 분석 결과"""
    stage: int          # 1 / 2 / 3
    stage_name: str
    scores: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    overall_score: float = 0.0


@dataclass
class PopupFullResult:
    """팝업 전체 분석 결과 (3단계 합산)"""
    action: str = "팝업(Pop-up) 3단계 분석"
    stages: list = field(default_factory=list)   # PopupStageResult 3개
    scores: dict = field(default_factory=dict)   # 단계별 점수 요약
    issues: list = field(default_factory=list)   # 전체 문제점
    overall_score: float = 0.0


# ──────────────────────────────────────────────
# 공통 헬퍼
# ──────────────────────────────────────────────

def _is_looking_down(kps: list[KeyPoint]) -> tuple[bool, float]:
    """
    시선이 아래를 향하는지 판단
    코(0)와 귀(3,4) 위치로 고개 숙임 여부 판단
    이미지 좌표계: y가 아래로 증가

    고개를 들면: 코 y < 귀 y (코가 귀보다 위)
    고개를 숙이면: 코 y > 귀 y (코가 귀보다 아래)
    """
    nose = kps[0]
    l_ear, r_ear = kps[3], kps[4]
    l_shoulder, r_shoulder = kps[5], kps[6]

    # 귀가 보이면 귀 기준으로 (더 정확)
    if _visible(nose, l_ear, r_ear, threshold=0.3):
        ear_y = (l_ear.y + r_ear.y) / 2
        diff = nose.y - ear_y  # 양수 = 코가 귀보다 아래 = 고개 숙임
        looking_down = diff > 10  # 10px 이상 아래면 고개 숙인 것
        return looking_down, round(diff, 1)

    # 귀가 안 보이면 어깨 기준
    if _visible(nose, l_shoulder, r_shoulder):
        shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
        diff = nose.y - shoulder_y
        # 어깨보다 코가 살짝이라도 아래면 고개 숙인 것 (민감도 강화)
        looking_down = diff > -5
        return looking_down, round(diff, 1)

    return False, 0.0


def _hand_position_ratio(kps: list[KeyPoint]) -> tuple[float, bool]:
    """
    손목이 어깨-엉덩이 사이(갈비뼈 옆)에 있는지 확인
    이상적: 손목 y좌표가 어깨와 엉덩이 y좌표 사이
    반환: (비율 0~1, 정상여부)
    """
    l_shoulder, r_shoulder = kps[5], kps[6]
    l_hip, r_hip = kps[11], kps[12]
    l_wrist, r_wrist = kps[9], kps[10]

    if not _visible(l_shoulder, r_shoulder, l_hip, r_hip, l_wrist, r_wrist):
        return 0.5, True

    shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
    hip_y = (l_hip.y + r_hip.y) / 2
    wrist_y = (l_wrist.y + r_wrist.y) / 2

    torso_height = hip_y - shoulder_y + 1e-6
    # 0 = 어깨 위치, 1 = 엉덩이 위치
    ratio = (wrist_y - shoulder_y) / torso_height
    # 이상적 범위: 0.3 ~ 0.7 (갈비뼈 옆)
    in_range = 0.25 <= ratio <= 0.75
    return round(ratio, 2), in_range


def _knee_angle_avg(kps: list[KeyPoint]) -> float:
    """양쪽 무릎 각도 평균"""
    l_hip, l_knee, l_ankle = kps[11], kps[13], kps[15]
    r_hip, r_knee, r_ankle = kps[12], kps[14], kps[16]
    angles = []
    if _visible(l_hip, l_knee, l_ankle):
        angles.append(_angle(l_hip, l_knee, l_ankle))
    if _visible(r_hip, r_knee, r_ankle):
        angles.append(_angle(r_hip, r_knee, r_ankle))
    return round(sum(angles) / len(angles), 1) if angles else 180.0


def _hip_height_ratio(kps: list[KeyPoint]) -> float:
    """
    엉덩이 높이 비율 (어깨 대비)
    낮을수록 쭈그린 자세
    0에 가까울수록 완전히 쭈그림
    """
    l_shoulder, r_shoulder = kps[5], kps[6]
    l_hip, r_hip = kps[11], kps[12]
    l_ankle, r_ankle = kps[15], kps[16]

    if not _visible(l_shoulder, r_shoulder, l_hip, r_hip, l_ankle, r_ankle):
        return 0.5

    shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
    hip_y = (l_hip.y + r_hip.y) / 2
    ankle_y = (l_ankle.y + r_ankle.y) / 2
    body_height = ankle_y - shoulder_y + 1e-6
    ratio = (hip_y - shoulder_y) / body_height
    return round(ratio, 2)


# ──────────────────────────────────────────────
# 1단계: 푸쉬(Push)
# ──────────────────────────────────────────────

def analyze_push_stage(kps: list[KeyPoint]) -> PopupStageResult:
    """
    체크포인트:
    - 손목이 갈비뼈 옆(어깨-엉덩이 사이 30~70%)에 위치
    - 팔은 완전히 펴야 함 (160~180도 이상적)
      → 팔을 쭉 펴야 체중이 뒤로 이동하고 테일에 양력이 최대로 발생
      → 팔을 구부리면 중심이 불안정해짐
    - 체중이 뒤로 이동 (상체가 뒤로 기울어지는지)
    """
    result = PopupStageResult(stage=1, stage_name="푸쉬(Push)")
    scores = {}
    issues = []

    # 1) 손 위치 (갈비뼈 옆인지)
    hand_ratio, in_range = _hand_position_ratio(kps)
    scores["손_위치_비율"] = hand_ratio
    if in_range:
        scores["손_위치_점수"] = 100
    elif hand_ratio < 0.25:
        issues.append("[푸쉬] 손이 너무 위(어깨 근처)에 있습니다. 갈비뼈 옆에 손을 위치하세요.")
        scores["손_위치_점수"] = max(0, round(hand_ratio / 0.25 * 70))
    else:
        issues.append("[푸쉬] 손이 너무 아래(엉덩이 근처)에 있습니다. 갈비뼈 옆에 손을 위치해야 팔로만 밀 수 있습니다.")
        scores["손_위치_점수"] = max(0, round(100 - (hand_ratio - 0.75) * 200))

    # 2) 팔 펴짐 확인 (완전히 펴야 함 — 160~180도 이상적)
    l_shoulder, l_elbow, l_wrist = kps[5], kps[7], kps[9]
    r_shoulder, r_elbow, r_wrist = kps[6], kps[8], kps[10]
    elbow_angles = []
    if _visible(l_shoulder, l_elbow, l_wrist):
        elbow_angles.append(_angle(l_shoulder, l_elbow, l_wrist))
    if _visible(r_shoulder, r_elbow, r_wrist):
        elbow_angles.append(_angle(r_shoulder, r_elbow, r_wrist))

    if elbow_angles:
        elbow_angle = sum(elbow_angles) / len(elbow_angles)
        scores["팔꿈치_각도"] = round(elbow_angle, 1)
        if elbow_angle >= 155:
            # 팔이 충분히 펴진 상태 — 체중이 뒤로 이동 가능
            scores["팔_펴짐_점수"] = 100
        elif elbow_angle >= 130:
            issues.append("[푸쉬] 팔을 더 펴야 합니다. 팔을 완전히 쭉 펴야 체중이 뒤로 이동하고 테일에 양력이 발생합니다.")
            scores["팔_펴짐_점수"] = max(0, round((elbow_angle - 100) / 55 * 80))
        else:
            issues.append("[푸쉬] 팔이 너무 구부러져 있습니다. 팔을 완전히 쭉 펴서 체중을 뒤쪽 테일로 보내야 파도를 잡을 수 있습니다.")
            scores["팔_펴짐_점수"] = max(0, round(elbow_angle / 130 * 50))

    score_vals = [v for k, v in scores.items() if k.endswith("_점수")]
    result.scores = scores
    result.issues = issues
    result.overall_score = round(sum(score_vals) / len(score_vals), 1) if score_vals else 0.0
    return result


# ──────────────────────────────────────────────
# 2단계: 발 끌어오기 & 쭈그리기(Pull & Squat)
# ──────────────────────────────────────────────

def analyze_squat_stage(kps: list[KeyPoint]) -> PopupStageResult:
    """
    체크포인트:
    - 무릎이 당겨지며 무릎 각도 감소 (90~130도 이상적)
    - 시선이 아래(바닥)를 보면 안 됨 ← 핵심!
    - 엉덩이 높이가 낮아진 상태
    """
    result = PopupStageResult(stage=2, stage_name="발 끌어오기(Pull & Squat)")
    scores = {}
    issues = []

    # 1) 시선 방향 ← 가장 중요
    looking_down, diff = _is_looking_down(kps)
    scores["시선_하향_정도"] = diff
    if looking_down:
        issues.append("[발 끌어오기] 바닥을 보고 있습니다! 시선을 파도 진행 방향으로 향하세요. 바닥을 보면 중심을 잃고 바로 빠집니다.")
        scores["시선_점수"] = 30
    else:
        scores["시선_점수"] = 100

    # 2) 무릎 당겨짐 (각도 감소)
    knee_angle = _knee_angle_avg(kps)
    scores["무릎_각도"] = knee_angle
    if knee_angle < 70:
        issues.append("[발 끌어오기] 무릎이 너무 많이 당겨졌습니다. 90~130도를 유지하세요.")
        scores["무릎_점수"] = max(0, round(knee_angle / 70 * 70))
    elif knee_angle > 160:
        issues.append("[발 끌어오기] 무릎이 충분히 당겨지지 않았습니다. 발을 몸 쪽으로 더 끌어오세요.")
        scores["무릎_점수"] = max(0, round(100 - (knee_angle - 130) * 2))
    else:
        scores["무릎_점수"] = 100

    # 3) 쭈그리기 정도
    # hip_ratio: 낮을수록 엉덩이가 높음(안 앉음), 높을수록 엉덩이가 낮음(많이 앉음)
    # 이상적: 0.55~0.70 (충분히 앉은 상태)
    hip_ratio = _hip_height_ratio(kps)
    scores["엉덩이_높이_비율"] = hip_ratio
    if hip_ratio < 0.45:
        issues.append("[발 끌어오기] 엉덩이가 너무 높습니다. 발을 끌어오며 제대로 앉는 자세가 필요합니다. 급하게 일어나면 머리가 앞뒤로 흔들려 중심을 잃습니다.")
        scores["쭈그리기_점수"] = max(0, round(hip_ratio / 0.45 * 60))
    elif hip_ratio > 0.80:
        issues.append("[발 끌어오기] 엉덩이가 너무 낮습니다. 허리가 과도하게 굽혀질 수 있습니다.")
        scores["쭈그리기_점수"] = max(0, round(100 - (hip_ratio - 0.75) * 200))
    else:
        scores["쭈그리기_점수"] = 100

    score_vals = [v for k, v in scores.items() if k.endswith("_점수")]
    result.scores = scores
    result.issues = issues
    result.overall_score = round(sum(score_vals) / len(score_vals), 1) if score_vals else 0.0
    return result


# ──────────────────────────────────────────────
# 3단계: 일어서기(Stand Up)
# ──────────────────────────────────────────────

def analyze_standup_stage(kps: list[KeyPoint]) -> PopupStageResult:
    """
    체크포인트:
    - 일어나는 동안 시선이 파도 방향을 향하는지 ← 핵심!
    - 무릎 각도 90~120도 (안정적 라이딩 자세)
    - 상체가 너무 앞으로 숙여지지 않는지
    """
    result = PopupStageResult(stage=3, stage_name="일어서기(Stand Up)")
    scores = {}
    issues = []

    # 1) 시선 방향 ← 가장 중요
    looking_down, diff = _is_looking_down(kps)
    scores["시선_하향_정도"] = diff
    if looking_down:
        issues.append("[일어서기] 일어나는 도중 바닥을 보고 있습니다! 시선을 진행 방향으로 고정하세요. 바닥을 보는 순간 무게중심이 앞으로 쏠려 빠집니다.")
        scores["시선_점수"] = 20  # 일어서기 단계에서는 더 중요하므로 감점 크게
    else:
        scores["시선_점수"] = 100

    # 2) 무릎 각도 (90~120도 이상적)
    knee_angle = _knee_angle_avg(kps)
    scores["무릎_각도"] = knee_angle
    if 80 <= knee_angle <= 130:
        scores["무릎_점수"] = 100
    elif knee_angle < 80:
        issues.append("[일어서기] 무릎을 너무 많이 구부렸습니다. 90~120도를 유지하세요.")
        scores["무릎_점수"] = max(0, round(knee_angle / 80 * 80))
    else:
        issues.append("[일어서기] 무릎이 너무 펴졌습니다. 더 낮은 자세가 안정적입니다. 무릎을 살짝 더 구부리세요.")
        scores["무릎_점수"] = max(0, round(100 - (knee_angle - 130) * 1.5))

    # 3) 상체 기울기 (앞으로 너무 숙이지 않는지)
    nose = kps[0]
    l_hip, r_hip = kps[11], kps[12]
    l_shoulder, r_shoulder = kps[5], kps[6]
    if _visible(nose, l_shoulder, r_shoulder, l_hip, r_hip):
        shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
        hip_y = (l_hip.y + r_hip.y) / 2
        # 어깨가 엉덩이보다 많이 앞(아래)으로 기울면 과도한 숙임
        forward_lean = shoulder_y - hip_y  # 음수면 어깨가 엉덩이보다 위
        scores["상체_기울기"] = round(forward_lean, 1)
        if forward_lean > 30:
            issues.append("[일어서기] 상체가 너무 앞으로 숙여졌습니다. 등을 세우고 가슴을 펴세요.")
            scores["상체_점수"] = max(0, round(100 - forward_lean))
        else:
            scores["상체_점수"] = 100

    score_vals = [v for k, v in scores.items() if k.endswith("_점수")]
    result.scores = scores
    result.issues = issues
    result.overall_score = round(sum(score_vals) / len(score_vals), 1) if score_vals else 0.0
    return result


# ──────────────────────────────────────────────
# 통합 — 프레임 리스트로 3단계 자동 감지
# ──────────────────────────────────────────────

def analyze_popup_stages(frames_keypoints: list[list]) -> PopupFullResult:
    """
    여러 프레임의 키포인트를 받아 3단계를 자동으로 감지하고 분석

    Args:
        frames_keypoints: 프레임별 키포인트 리스트
                          [[kp_frame1], [kp_frame2], ...]
                          각 kp는 [[x,y,conf]*17]

    Returns:
        PopupFullResult (3단계 결과 포함)
    """
    n = len(frames_keypoints)
    if n == 0:
        raise ValueError("프레임이 없습니다.")

    # 프레임 수에 따라 3단계 구간 자동 분할
    # 앞 33% → 푸쉬, 중간 33% → 발 끌어오기, 뒤 33% → 일어서기
    def _best_kps(start_ratio: float, end_ratio: float) -> list[KeyPoint]:
        start = max(0, int(n * start_ratio))
        end = min(n, int(n * end_ratio))
        segment = frames_keypoints[start:end] if end > start else frames_keypoints[start:start+1]
        # 평균 신뢰도가 가장 높은 프레임 선택
        best = max(segment, key=lambda kp: sum(p[2] for p in kp) / len(kp))
        return [KeyPoint(x=p[0], y=p[1], confidence=p[2]) for p in best]

    kps_push   = _best_kps(0.0, 0.33)
    kps_squat  = _best_kps(0.33, 0.66)
    kps_standup = _best_kps(0.66, 1.0)

    stage1 = analyze_push_stage(kps_push)
    stage2 = analyze_squat_stage(kps_squat)
    stage3 = analyze_standup_stage(kps_standup)

    all_issues = stage1.issues + stage2.issues + stage3.issues
    overall = round((stage1.overall_score + stage2.overall_score + stage3.overall_score) / 3, 1)

    scores_summary = {
        "1단계_푸쉬_점수": stage1.overall_score,
        "2단계_발끌어오기_점수": stage2.overall_score,
        "3단계_일어서기_점수": stage3.overall_score,
    }

    return PopupFullResult(
        stages=[stage1, stage2, stage3],
        scores=scores_summary,
        issues=all_issues,
        overall_score=overall,
    )
