"""
서핑 자세 분석 모듈

YOLOv8-pose 가 추출한 17개 키포인트 좌표를 받아
테이크오프 / 스탠스 / 패들링 각각의 체크포인트를 계산하고
점수 + 문제점 리스트를 반환한다.

키포인트 인덱스 (COCO 17-keypoint 기준):
  0: 코     1: 왼눈   2: 오른눈  3: 왼귀   4: 오른귀
  5: 왼어깨  6: 오른어깨 7: 왼팔꿈치 8: 오른팔꿈치
  9: 왼손목 10: 오른손목 11: 왼엉덩이 12: 오른엉덩이
 13: 왼무릎 14: 오른무릎 15: 왼발목 16: 오른발목
"""

import math
import numpy as np
from dataclasses import dataclass, field


@dataclass
class KeyPoint:
    x: float
    y: float
    confidence: float = 1.0


@dataclass
class AnalysisResult:
    action: str
    scores: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    overall_score: float = 0.0


# ──────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────

def _angle(a: KeyPoint, b: KeyPoint, c: KeyPoint) -> float:
    """세 점 a-b-c 에서 b를 꼭짓점으로 하는 각도(도) 반환"""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))


def _distance(a: KeyPoint, b: KeyPoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _visible(*kps: KeyPoint, threshold: float = 0.5) -> bool:
    return all(k.confidence >= threshold for k in kps)


# ──────────────────────────────────────────────
# 테이크오프 분석
# ──────────────────────────────────────────────

def analyze_takeoff(kps: list[KeyPoint]) -> AnalysisResult:
    """
    체크포인트:
    - 무릎 굽힘 각도 (이상: 90~120도)
    - 손 위치 (가슴 옆에 있는지)
    - 시선 방향 (코가 어깨보다 위를 향하는지)
    """
    result = AnalysisResult(action="테이크오프(Take-off)")
    scores = {}
    issues = []

    # 1) 무릎 굽힘 각도 (왼쪽 기준: 엉덩이-무릎-발목)
    l_hip, l_knee, l_ankle = kps[11], kps[13], kps[15]
    r_hip, r_knee, r_ankle = kps[12], kps[14], kps[16]

    if _visible(l_hip, l_knee, l_ankle):
        knee_angle = _angle(l_hip, l_knee, l_ankle)
        scores["무릎_굽힘_각도"] = round(knee_angle, 1)
        if knee_angle < 80:
            issues.append("무릎을 너무 많이 구부렸습니다. 90~120도를 유지하세요.")
        elif knee_angle > 150:
            issues.append("무릎이 거의 펴져 있습니다. 더 낮은 자세가 필요합니다.")
        knee_score = 100 - abs(knee_angle - 105) * 1.5
        scores["무릎_점수"] = max(0, min(100, round(knee_score)))
    elif _visible(r_hip, r_knee, r_ankle):
        knee_angle = _angle(r_hip, r_knee, r_ankle)
        scores["무릎_굽힘_각도"] = round(knee_angle, 1)
        knee_score = 100 - abs(knee_angle - 105) * 1.5
        scores["무릎_점수"] = max(0, min(100, round(knee_score)))

    # 2) 시선 방향 (코 y좌표 vs 어깨 y좌표 — 이미지 좌표계에서 y↓)
    nose = kps[0]
    l_shoulder, r_shoulder = kps[5], kps[6]
    if _visible(nose, l_shoulder, r_shoulder):
        shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
        if nose.y > shoulder_y * 1.05:
            issues.append("시선이 아래를 향하고 있습니다. 진행 방향을 바라보세요.")
            scores["시선_점수"] = 50
        else:
            scores["시선_점수"] = 100

    # 3) 손 위치 (손목이 어깨보다 안쪽에 있는지)
    l_wrist, r_wrist = kps[9], kps[10]
    if _visible(l_shoulder, r_shoulder, l_wrist, r_wrist):
        shoulder_width = _distance(l_shoulder, r_shoulder)
        wrist_width = _distance(l_wrist, r_wrist)
        ratio = wrist_width / (shoulder_width + 1e-6)
        scores["손_위치_비율"] = round(ratio, 2)
        if ratio > 1.5:
            issues.append("손이 너무 벌어졌습니다. 가슴 옆에 손을 위치하세요.")
            scores["손_점수"] = 60
        else:
            scores["손_점수"] = 100

    score_vals = [v for k, v in scores.items() if k.endswith("_점수")]
    result.scores = scores
    result.issues = issues
    result.overall_score = round(sum(score_vals) / len(score_vals), 1) if score_vals else 0.0
    return result


# ──────────────────────────────────────────────
# 스탠스 분석
# ──────────────────────────────────────────────

def analyze_stance(kps: list[KeyPoint]) -> AnalysisResult:
    """
    체크포인트:
    - 발 간격 (어깨너비 기준, 이상: 1.0~1.5배)
    - 무게중심 (앞발 60%, 뒷발 40% — 발목 y 비교)
    - 어깨 정렬
    """
    result = AnalysisResult(action="스탠스(Stance)")
    scores = {}
    issues = []

    l_shoulder, r_shoulder = kps[5], kps[6]
    l_ankle, r_ankle = kps[15], kps[16]

    # 1) 발 간격
    if _visible(l_shoulder, r_shoulder, l_ankle, r_ankle):
        shoulder_width = _distance(l_shoulder, r_shoulder)
        foot_gap = _distance(l_ankle, r_ankle)
        ratio = foot_gap / (shoulder_width + 1e-6)
        scores["발간격_어깨너비_비율"] = round(ratio, 2)

        if ratio < 0.8:
            issues.append("발 간격이 너무 좁습니다. 어깨너비로 벌려주세요.")
            scores["발간격_점수"] = max(0, round(ratio / 0.8 * 80))
        elif ratio > 2.0:
            issues.append("발 간격이 너무 넓습니다. 어깨너비 1~1.5배를 유지하세요.")
            scores["발간격_점수"] = max(0, round(100 - (ratio - 2.0) * 30))
        else:
            scores["발간격_점수"] = 100

    # 2) 무게중심 (앞발 무릎 각도로 추정)
    l_hip, r_hip = kps[11], kps[12]
    l_knee, r_knee = kps[13], kps[14]
    if _visible(l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle):
        l_knee_angle = _angle(l_hip, l_knee, l_ankle)
        r_knee_angle = _angle(r_hip, r_knee, r_ankle)
        bend_diff = abs(l_knee_angle - r_knee_angle)
        scores["좌우_무릎각도_차이"] = round(bend_diff, 1)
        if bend_diff > 30:
            issues.append("좌우 무게중심이 불균형합니다. 앞발에 60% 체중을 실어보세요.")
            scores["무게중심_점수"] = max(0, round(100 - bend_diff))
        else:
            scores["무게중심_점수"] = 100

    score_vals = [v for k, v in scores.items() if k.endswith("_점수")]
    result.scores = scores
    result.issues = issues
    result.overall_score = round(sum(score_vals) / len(score_vals), 1) if score_vals else 0.0
    return result


# ──────────────────────────────────────────────
# 패들링 분석
# ──────────────────────────────────────────────

def analyze_paddling(kps: list[KeyPoint]) -> AnalysisResult:
    """
    패들링 기술 원리 (LLM 피드백 참고용):
      - 올바른 패들: 팔꿈치를 수직으로 물속 깊이 넣고 손바닥으로 물을 뒤로 밀어냄
        (스케이트보드에 엎드려 손바닥으로 바닥을 미는 느낌 / 최대한 많은 물을 뒤로)
      - 잘못된 패들 (퍼올리기): 팔꿈치가 수직으로 안 들어가고 물을 퍼올리는 동작
        → 추진력이 아닌 뒤뚱거리는 방향의 힘 발생 → 불안정성 증가
      - 팔꿈치 높이(High Elbow)는 팔을 물 밖으로 꺼내 돌아오는 리커버리 단계에서만 확인 가능
        → 단일 프레임으로는 스트로크/리커버리 단계 구분이 어려워 직접 체크 불가

    YOLO로 측정 가능한 항목:
    - 몸통 위치 (좌우 어깨 대칭 — 보드 중심 유지 여부)
    - 팔 스트로크 대칭 (좌우 팔꿈치 높이 차이)
    - 머리 위치 (과도하게 들지 않는지)
    - 어깨 앞쪽 뻗음 (입수 준비 자세)
    """
    result = AnalysisResult(action="패들링(Paddling)")
    scores = {}
    issues = []

    l_shoulder, r_shoulder = kps[5], kps[6]
    l_elbow, r_elbow = kps[7], kps[8]
    l_wrist, r_wrist = kps[9], kps[10]
    nose = kps[0]

    # 1) 좌우 어깨 대칭 (보드 중심 유지 여부)
    if _visible(l_shoulder, r_shoulder):
        shoulder_y_diff = abs(l_shoulder.y - r_shoulder.y)
        shoulder_width = _distance(l_shoulder, r_shoulder)
        tilt_ratio = shoulder_y_diff / (shoulder_width + 1e-6)
        scores["어깨_기울기"] = round(tilt_ratio, 3)
        if tilt_ratio > 0.15:
            issues.append("[패들링] 몸이 한쪽으로 기울었습니다. 보드 중심에 눕는 자세를 유지하세요.")
            scores["몸통_대칭_점수"] = max(0, round(100 - tilt_ratio * 300))
        else:
            scores["몸통_대칭_점수"] = 100

    # 2) 팔 스트로크 좌우 대칭
    # 한쪽 팔이 스트로크 중, 반대쪽이 리커버리 중이므로 팔꿈치 높이 차이가 너무 크면 문제
    if _visible(l_elbow, r_elbow, threshold=0.3):
        elbow_y_diff = abs(l_elbow.y - r_elbow.y)
        scores["팔꿈치_높이_차이_px"] = round(elbow_y_diff, 1)
        if elbow_y_diff > 60:
            issues.append("[패들링] 좌우 패들링 스트로크가 불균형합니다. 양팔을 번갈아 균등하게 저으세요.")
            scores["패들링_대칭_점수"] = max(0, round(100 - elbow_y_diff * 0.7))
        else:
            scores["패들링_대칭_점수"] = 100

    # 3) 팔 뻗음 (입수 전 팔이 충분히 앞으로 뻗어있는지)
    # 손목이 어깨보다 앞으로(x좌표 기준) 나와있어야 물을 깊이 넣을 수 있음
    reach_checks = []
    if _visible(l_shoulder, l_wrist, threshold=0.3):
        # x좌표: 손목이 어깨보다 앞에 있는지 (방향에 따라 다르므로 절대 거리로)
        l_reach = abs(l_wrist.x - l_shoulder.x)
        reach_checks.append(l_reach)
        scores["왼팔_뻗음_px"] = round(l_reach, 1)
    if _visible(r_shoulder, r_wrist, threshold=0.3):
        r_reach = abs(r_wrist.x - r_shoulder.x)
        reach_checks.append(r_reach)
        scores["오른팔_뻗음_px"] = round(r_reach, 1)

    if reach_checks:
        avg_reach = sum(reach_checks) / len(reach_checks)
        shoulder_width = _distance(l_shoulder, r_shoulder) if _visible(l_shoulder, r_shoulder) else 100
        reach_ratio = avg_reach / (shoulder_width + 1e-6)
        scores["팔뻗음_비율"] = round(reach_ratio, 2)
        if reach_ratio < 0.3:
            issues.append("[패들링] 팔을 충분히 앞으로 뻗지 않고 있습니다. 팔꿈치를 수직으로 깊이 넣기 위해 먼저 팔을 앞으로 충분히 뻗어야 합니다.")
            scores["팔뻗음_점수"] = max(0, round(reach_ratio / 0.3 * 70))
        else:
            scores["팔뻗음_점수"] = 100

    # 4) 머리 과도한 들기
    if _visible(nose, l_shoulder, r_shoulder):
        shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
        head_lift = shoulder_y - nose.y  # 양수 = 머리가 위
        scores["머리_들기_정도_px"] = round(head_lift, 1)
        if head_lift > 80:
            issues.append("[패들링] 머리를 너무 높이 들고 있습니다. 목에 힘을 빼고 자연스럽게 유지하세요.")
            scores["머리_자세_점수"] = max(0, round(100 - (head_lift - 80) * 0.8))
        else:
            scores["머리_자세_점수"] = 100

    score_vals = [v for k, v in scores.items() if k.endswith("_점수")]
    result.scores = scores
    result.issues = issues
    result.overall_score = round(sum(score_vals) / len(score_vals), 1) if score_vals else 0.0
    return result


# ──────────────────────────────────────────────
# 통합 진입점
# ──────────────────────────────────────────────

ACTION_MAP = {
    "takeoff": analyze_takeoff,
    "stance": analyze_stance,
    "paddling": analyze_paddling,
}


def analyze(action: str, keypoints_raw: list) -> AnalysisResult:
    """
    action: "takeoff" | "stance" | "paddling"
    keypoints_raw: YOLOv8 출력 형식 [[x, y, conf], ...] 17개
    """
    if action not in ACTION_MAP:
        raise ValueError(f"지원하지 않는 동작: {action}. 가능한 값: {list(ACTION_MAP)}")

    kps = [KeyPoint(x=p[0], y=p[1], confidence=p[2]) for p in keypoints_raw]
    return ACTION_MAP[action](kps)
