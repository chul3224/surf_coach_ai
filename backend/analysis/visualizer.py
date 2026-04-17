"""
서핑 자세 시각화 모듈

YOLOv8 키포인트 + 자세 분석 점수를 받아
관절별 색상 오버레이 이미지를 생성한다.

점수 색상 기준:
  70~100점 → 초록색
  50~70점  → 주황색
  0~50점   → 빨간색
"""

import cv2
import numpy as np
from pathlib import Path


# COCO 17 키포인트 연결선 (스켈레톤)
SKELETON = [
    (0, 1), (0, 2),           # 코 → 눈
    (1, 3), (2, 4),           # 눈 → 귀
    (5, 6),                   # 어깨 사이
    (5, 7), (7, 9),           # 왼팔
    (6, 8), (8, 10),          # 오른팔
    (5, 11), (6, 12),         # 어깨 → 엉덩이
    (11, 12),                 # 엉덩이 사이
    (11, 13), (13, 15),       # 왼다리
    (12, 14), (14, 16),       # 오른다리
]

# 키포인트 인덱스별 이름
KP_NAMES = [
    "코", "왼눈", "오른눈", "왼귀", "오른귀",
    "왼어깨", "오른어깨", "왼팔꿈치", "오른팔꿈치",
    "왼손목", "오른손목", "왼엉덩이", "오른엉덩이",
    "왼무릎", "오른무릎", "왼발목", "오른발목",
]

# 동작별로 점수와 연관된 주요 키포인트 매핑
ACTION_KP_SCORE_MAP = {
    "takeoff": {
        "무릎_점수":    [13, 14],   # 왼무릎, 오른무릎
        "시선_점수":    [0],         # 코
        "손_점수":      [9, 10],     # 손목
    },
    "stance": {
        "발간격_점수":  [15, 16],    # 발목
        "무게중심_점수": [11, 12, 13, 14],
    },
    "paddling": {
        "몸통_대칭_점수":   [5, 6],       # 어깨
        "패들링_대칭_점수": [7, 8],       # 팔꿈치
        "팔뻗음_점수":      [9, 10],      # 손목
        "머리_자세_점수":   [0],           # 코
    },
}


def _score_to_color(score: float) -> tuple:
    """점수 → BGR 색상"""
    if score >= 70:
        return (0, 200, 0)      # 초록
    elif score >= 50:
        return (0, 140, 255)    # 주황
    else:
        return (0, 0, 220)      # 빨강


def _kp_colors(action: str, scores: dict) -> list:
    """
    17개 키포인트 각각의 색상 결정

    점수가 매핑된 키포인트는 해당 점수 색상,
    나머지는 전체 overall_score 색상 사용
    """
    overall = scores.get("overall_score", 100)
    colors = [_score_to_color(overall)] * 17

    mapping = ACTION_KP_SCORE_MAP.get(action, {})
    for score_key, kp_indices in mapping.items():
        score_val = scores.get(score_key, overall)
        color = _score_to_color(score_val)
        for idx in kp_indices:
            colors[idx] = color

    return colors


def draw_overlay(
    frame: np.ndarray,
    keypoints: list,
    action: str,
    scores: dict,
    overall_score: float,
) -> np.ndarray:
    """
    프레임 위에 관절 오버레이 그리기

    Args:
        frame: BGR 이미지 (numpy array)
        keypoints: [[x, y, conf], ...] 17개
        action: "takeoff" | "stance" | "paddling"
        scores: 세부 점수 dict
        overall_score: 종합 점수

    Returns:
        오버레이가 그려진 BGR 이미지
    """
    img = frame.copy()
    scores_with_overall = {**scores, "overall_score": overall_score}
    colors = _kp_colors(action, scores_with_overall)

    # --- 스켈레톤 연결선 ---
    for (i, j) in SKELETON:
        kp_i = keypoints[i]
        kp_j = keypoints[j]
        if kp_i[2] < 0.3 or kp_j[2] < 0.3:
            continue
        pt1 = (int(kp_i[0]), int(kp_i[1]))
        pt2 = (int(kp_j[0]), int(kp_j[1]))
        # 선 색상은 두 관절 중 낮은 점수 기준
        line_color = colors[i] if colors[i] < colors[j] else colors[j]
        cv2.line(img, pt1, pt2, line_color, 2, cv2.LINE_AA)

    # --- 관절 점 + 이름 ---
    for idx, kp in enumerate(keypoints):
        x, y, conf = kp
        if conf < 0.3:
            continue
        cx, cy = int(x), int(y)
        color = colors[idx]

        # 관절 원
        cv2.circle(img, (cx, cy), 6, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)  # 흰 테두리

        # 관절 이름 (작게)
        cv2.putText(
            img, KP_NAMES[idx],
            (cx + 8, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    # --- 점수 패널 (우상단) ---
    _draw_score_panel(img, action, overall_score, scores)

    return img


def _draw_score_panel(img, action: str, overall_score: float, scores: dict):
    """우상단에 점수 패널 표시"""
    h, w = img.shape[:2]
    panel_x = w - 230
    panel_y = 10

    # 반투명 배경
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x - 10, panel_y), (w - 5, panel_y + 160), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # 동작명 (OpenCV 기본 폰트는 한글 미지원 → 영문 표기)
    action_en = {"takeoff": "Take-off", "stance": "Stance", "paddling": "Paddling"}.get(action, action)
    cv2.putText(img, f"[{action_en}]", (panel_x, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # 종합 점수
    score_color = _score_to_color(overall_score)
    cv2.putText(img, f"Score: {overall_score:.1f}", (panel_x, panel_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2, cv2.LINE_AA)

    # 범례
    y_offset = panel_y + 80
    for label, color in [("70~100 Good", (0, 200, 0)), ("50~70  Check", (0, 140, 255)), ("~50   Fix", (0, 0, 220))]:
        cv2.circle(img, (panel_x, y_offset), 5, color, -1)
        cv2.putText(img, label, (panel_x + 12, y_offset + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        y_offset += 22


def save_overlay_image(
    frame: np.ndarray,
    keypoints: list,
    action: str,
    scores: dict,
    overall_score: float,
    save_path: str,
) -> str:
    """오버레이 이미지를 저장하고 경로 반환"""
    result_img = draw_overlay(frame, keypoints, action, scores, overall_score)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, result_img)
    return save_path
