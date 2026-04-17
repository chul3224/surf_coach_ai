"""
레퍼런스 포즈 매칭 시스템

data/reference/ 폴더의 사진들로부터 동작별 평균 키포인트를 계산하고,
사용자 프레임의 키포인트와 코사인 유사도를 비교해 가장 가까운 동작을 분류한다.

폴더 구조:
    data/reference/
        takeoff_push/      ← 푸쉬 동작 범위 프레임들
        takeoff_squat/     ← 발 끌어오기 범위 프레임들
        takeoff_standup/   ← 일어서기 범위 프레임들
        stance/            ← 스탠스 프레임들
        paddling/          ← 패들링 프레임들

사용법:
    # 최초 1회: 레퍼런스 빌드 (JSON 저장)
    build_reference_db()

    # 이후: 사용자 키포인트 → 동작 분류
    action, scores = match_pose(user_keypoints)
"""

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# 레퍼런스 폴더 경로
REFERENCE_DIR = Path(__file__).parent.parent.parent / "data" / "reference"
REFERENCE_DB_PATH = Path(__file__).parent.parent.parent / "data" / "reference_poses.json"

# 폴더명 → 동작 레이블
FOLDER_TO_LABEL = {
    "takeoff_push":    "takeoff_push",
    "takeoff_squat":   "takeoff_squat",
    "takeoff_standup": "takeoff_standup",
    "stance":          "stance",
    "paddling":        "paddling",
}

# 신뢰도 임계값
CONF_THRESHOLD = 0.3

_model = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8n-pose.pt")
    return _model


def _extract_keypoints_from_image(image_path: str) -> list | None:
    """
    이미지 한 장에서 키포인트 추출
    Returns: [[x, y, conf], ...] 17개 or None
    """
    model = _get_model()
    img = cv2.imread(image_path)
    if img is None:
        return None

    # conf=0.25로 낮춰서 엎드린 자세(패들링 등)도 감지
    results = model(img, verbose=False, conf=0.25)
    if not results or results[0].keypoints is None:
        return None

    kps_data = results[0].keypoints
    if kps_data.xy is None or len(kps_data.xy) == 0:
        return None

    # 바운딩박스 면적이 가장 큰 사람 선택
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_idx = 0
    max_area = -1
    for i, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = box.cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_idx = i

    if best_idx >= len(kps_data.xy):
        return None

    xy = kps_data.xy[best_idx].cpu().numpy()
    conf = kps_data.conf[best_idx].cpu().numpy()

    return [[float(xy[i][0]), float(xy[i][1]), float(conf[i])] for i in range(17)]


def _kps_to_vector(kps: list) -> np.ndarray | None:
    """
    키포인트 리스트 → 정규화된 특징 벡터
    - 신뢰도 낮은 관절 제외
    - 어깨 중심점 기준 정규화 (없으면 엉덩이 기준으로 대체 — 패들링 등 엎드린 자세 대응)
    - 기준축 너비로 스케일 정규화 (크기 불변)
    """
    kps = np.array(kps)  # (17, 3)
    xy = kps[:, :2]
    conf = kps[:, 2]

    # 기준점 결정: 어깨 우선, 없으면 엉덩이 사용
    l_sh_ok = conf[5] >= CONF_THRESHOLD
    r_sh_ok = conf[6] >= CONF_THRESHOLD
    l_hip_ok = conf[11] >= CONF_THRESHOLD
    r_hip_ok = conf[12] >= CONF_THRESHOLD

    if l_sh_ok and r_sh_ok:
        center = (xy[5] + xy[6]) / 2
        scale = np.linalg.norm(xy[5] - xy[6]) + 1e-6
    elif l_hip_ok and r_hip_ok:
        # 엉덩이 기준 (패들링처럼 어깨가 잘 안 보이는 경우)
        center = (xy[11] + xy[12]) / 2
        scale = np.linalg.norm(xy[11] - xy[12]) + 1e-6
    elif l_sh_ok or r_sh_ok:
        # 한쪽 어깨만 있는 경우
        anchor_idx = 5 if l_sh_ok else 6
        center = xy[anchor_idx]
        # 어깨-엉덩이 거리로 스케일
        hip_ok = [i for i in [11, 12] if conf[i] >= CONF_THRESHOLD]
        if hip_ok:
            scale = np.linalg.norm(xy[anchor_idx] - xy[hip_ok[0]]) + 1e-6
        else:
            return None
    else:
        return None

    if scale < 5:  # 스케일이 너무 작으면 신뢰 불가
        return None

    normalized = (xy - center) / scale  # (17, 2)

    # 신뢰도 가중치 적용 (낮은 conf 관절은 0으로)
    weights = np.where(conf >= CONF_THRESHOLD, conf, 0.0)
    weighted = normalized * weights[:, np.newaxis]  # (17, 2)

    return weighted.flatten()  # (34,)


def _average_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    """유효한 벡터들의 평균"""
    arr = np.array(vectors)
    return np.mean(arr, axis=0)


def build_reference_db(force: bool = False) -> dict:
    """
    레퍼런스 폴더의 사진들로 동작별 평균 키포인트 벡터 계산 후 JSON 저장

    Args:
        force: True면 기존 DB 무시하고 재빌드

    Returns:
        {label: {"mean_vector": [...], "sample_count": N, "image_count": N}}
    """
    if not force and REFERENCE_DB_PATH.exists():
        print(f"기존 레퍼런스 DB 로드: {REFERENCE_DB_PATH}")
        with open(REFERENCE_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print("레퍼런스 DB 빌드 시작...")
    db = {}

    for folder_name, label in FOLDER_TO_LABEL.items():
        folder_path = REFERENCE_DIR / folder_name
        if not folder_path.exists():
            print(f"  [경고] 폴더 없음: {folder_path}")
            continue

        image_files = sorted([
            p for p in folder_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ])

        if not image_files:
            print(f"  [경고] {folder_name}: 이미지 없음")
            continue

        vectors = []
        failed = 0
        for img_path in image_files:
            kps = _extract_keypoints_from_image(str(img_path))
            if kps is None:
                failed += 1
                continue
            vec = _kps_to_vector(kps)
            if vec is None:
                failed += 1
                continue
            vectors.append(vec)

        if not vectors:
            print(f"  [실패] {folder_name}: 유효한 키포인트 없음")
            continue

        mean_vec = _average_vectors(vectors)
        # L2 정규화 (코사인 유사도 계산 시 dot product = cosine similarity)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-6:
            mean_vec = mean_vec / norm

        db[label] = {
            "mean_vector": mean_vec.tolist(),
            "sample_count": len(vectors),
            "image_count": len(image_files),
            "failed_count": failed,
        }
        print(f"  ✅ {folder_name}: {len(vectors)}/{len(image_files)}장 성공")

    # JSON 저장
    REFERENCE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print(f"레퍼런스 DB 저장 완료: {REFERENCE_DB_PATH}")
    return db


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """코사인 유사도 (v2는 이미 L2 정규화된 상태)"""
    norm1 = np.linalg.norm(v1)
    if norm1 < 1e-6:
        return 0.0
    v1_norm = v1 / norm1
    return float(np.dot(v1_norm, v2))


_reference_db: dict | None = None


def _load_db() -> dict:
    global _reference_db
    if _reference_db is None:
        if not REFERENCE_DB_PATH.exists():
            _reference_db = build_reference_db()
        else:
            with open(REFERENCE_DB_PATH, "r", encoding="utf-8") as f:
                _reference_db = json.load(f)
    return _reference_db


def match_pose(keypoints: list) -> tuple[str, dict[str, float]]:
    """
    사용자 키포인트 → 가장 유사한 동작 레이블 반환

    Args:
        keypoints: [[x, y, conf], ...] 17개

    Returns:
        (best_label, scores)
        best_label: "takeoff_push" | "takeoff_squat" | "takeoff_standup" | "stance" | "paddling"
        scores: {"takeoff_push": 0.92, "stance": 0.78, ...}
    """
    db = _load_db()
    if not db:
        return "unknown", {}

    user_vec = _kps_to_vector(keypoints)
    if user_vec is None:
        return "unknown", {}

    scores = {}
    for label, data in db.items():
        ref_vec = np.array(data["mean_vector"])
        sim = _cosine_similarity(user_vec, ref_vec)
        # 유사도를 0~100 점수로 변환
        scores[label] = round(max(0.0, sim) * 100, 1)

    best_label = max(scores, key=lambda k: scores[k])
    return best_label, scores


def match_pose_for_frame_selection(
    keypoints_list: list[list],
    target_label: str,
) -> int:
    """
    여러 프레임 중 target_label에 가장 가까운 프레임 인덱스 반환
    (stance 분석 시 takeoff_standup 프레임 혼입 문제 해결용)

    Args:
        keypoints_list: 프레임별 키포인트 리스트
        target_label: 찾고 싶은 동작 ("stance", "paddling", ...)

    Returns:
        가장 유사한 프레임의 인덱스
    """
    db = _load_db()
    if not db or target_label not in db:
        # DB 없으면 신뢰도 가장 높은 프레임 반환
        return max(range(len(keypoints_list)),
                   key=lambda i: sum(p[2] for p in keypoints_list[i]) / 17)

    ref_vec = np.array(db[target_label]["mean_vector"])
    best_idx = 0
    best_sim = -1.0

    for i, kps in enumerate(keypoints_list):
        user_vec = _kps_to_vector(kps)
        if user_vec is None:
            continue
        sim = _cosine_similarity(user_vec, ref_vec)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    return best_idx


def get_reference_info() -> dict:
    """레퍼런스 DB 현황 조회"""
    db = _load_db()
    return {
        label: {
            "sample_count": data["sample_count"],
            "image_count": data["image_count"],
        }
        for label, data in db.items()
    }
