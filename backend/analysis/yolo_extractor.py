"""
YOLOv8-pose 키포인트 추출기

업로드된 영상에서 대표 프레임을 뽑고
YOLOv8-pose 로 17개 관절 좌표를 추출한다.

반환 형식: [[x, y, confidence], ...] 17개
"""

import math
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 모델은 처음 한 번만 로드 (싱글톤)
_model = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        # 없으면 자동 다운로드 (~7MB)
        _model = YOLO("yolov8n-pose.pt")
    return _model


def extract_keypoints_from_video(video_path: str) -> tuple[list, np.ndarray]:
    """
    영상에서 가장 잘 감지된 단일 프레임의 키포인트 반환
    (stance / paddling 분석용)

    Returns:
        keypoints: [[x, y, conf], ...] 17개
        frame: 해당 프레임 이미지 (BGR numpy array)
    """
    all_kps, frames = extract_multi_keypoints_from_video(video_path, num_samples=5)
    if not all_kps:
        raise ValueError("영상에서 사람을 감지하지 못했습니다. 서퍼가 잘 보이는 영상을 사용해주세요.")

    # 신뢰도 가장 높은 프레임 선택
    best_idx = max(range(len(all_kps)), key=lambda i: sum(p[2] for p in all_kps[i]) / 17)
    return all_kps[best_idx], frames[best_idx]


def extract_multi_keypoints_from_video(
    video_path: str,
    num_samples: int = 9,
) -> tuple[list[list], list[np.ndarray]]:
    """
    영상에서 팝업 동작 구간을 감지하여 여러 프레임의 키포인트 반환

    팝업 감지 전략:
    - 어깨 y좌표가 급격히 올라가는(숫자가 작아지는) 구간 = 일어나는 중
    - 전체 영상에서 어깨 높이 변화가 가장 큰 구간을 팝업으로 판단
    - 해당 구간을 3등분하여 각 단계 프레임 추출

    Returns:
        all_keypoints: 프레임별 키포인트 리스트 [[kp*17], ...]
        frames: 프레임 이미지 리스트
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model = _get_model()

    # 1단계: 전체 영상을 15프레임으로 스캔 → 어깨 높이 추적
    scan_count = min(15, total_frames)
    scan_indices = [int(total_frames * i / scan_count) for i in range(scan_count)]
    shoulder_heights = []  # (frame_idx, shoulder_y)

    for idx in scan_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            shoulder_heights.append((idx, None))
            continue

        results = model(frame, verbose=False)
        if not results or results[0].keypoints is None:
            shoulder_heights.append((idx, None))
            continue

        kps_data = results[0].keypoints
        if kps_data.xy is None or len(kps_data.xy) == 0:
            shoulder_heights.append((idx, None))
            continue

        person_idx = _pick_main_person(results[0])
        if person_idx < 0:
            shoulder_heights.append((idx, None))
            continue

        xy = kps_data.xy[person_idx].cpu().numpy()
        conf = kps_data.conf[person_idx].cpu().numpy()

        # 어깨 y좌표 평균 (낮을수록 화면 위 = 일어선 상태)
        l_sh_y = float(xy[5][1]) if conf[5] > 0.3 else None
        r_sh_y = float(xy[6][1]) if conf[6] > 0.3 else None
        if l_sh_y and r_sh_y:
            shoulder_heights.append((idx, (l_sh_y + r_sh_y) / 2))
        elif l_sh_y:
            shoulder_heights.append((idx, l_sh_y))
        elif r_sh_y:
            shoulder_heights.append((idx, r_sh_y))
        else:
            shoulder_heights.append((idx, None))

    # 2단계: 어깨가 가장 크게 올라가는 구간 찾기 (팝업 구간)
    valid = [(i, h) for i, (_, h) in enumerate(shoulder_heights) if h is not None]
    popup_start_scan = 0
    popup_end_scan = len(shoulder_heights) - 1

    if len(valid) >= 3:
        max_drop = 0
        for k in range(len(valid) - 2):
            drop = valid[k][1] - valid[k + 2][1]  # y가 줄어드는 것 = 올라가는 것
            if drop > max_drop:
                max_drop = drop
                popup_start_scan = valid[k][0]
                popup_end_scan = valid[min(k + 2, len(valid) - 1)][0]

    popup_start_frame = shoulder_heights[popup_start_scan][0]
    popup_end_frame = shoulder_heights[popup_end_scan][0]

    # 팝업 구간이 너무 짧으면 전체 영상 사용
    if popup_end_frame - popup_start_frame < total_frames * 0.1:
        popup_start_frame = 0
        popup_end_frame = total_frames - 1

    # 3단계: 팝업 구간을 num_samples 개로 균등 샘플링
    span = popup_end_frame - popup_start_frame
    sample_indices = sorted(set([
        max(0, int(popup_start_frame + span * i / max(num_samples - 1, 1)))
        for i in range(num_samples)
    ]))

    all_keypoints = []
    frames = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False)
        if not results or results[0].keypoints is None:
            continue

        kps_data = results[0].keypoints
        if kps_data.xy is None or len(kps_data.xy) == 0:
            continue

        person_idx = _pick_main_person(results[0])
        if person_idx < 0:
            continue

        xy = kps_data.xy[person_idx].cpu().numpy()
        conf = kps_data.conf[person_idx].cpu().numpy()

        kps = [
            [float(xy[i][0]), float(xy[i][1]), float(conf[i])]
            for i in range(17)
        ]
        all_keypoints.append(kps)
        frames.append(frame.copy())

    cap.release()
    return all_keypoints, frames


def _pick_main_person(result) -> int:
    """여러 명 감지 시 바운딩박스 면적이 가장 큰 사람 인덱스 반환"""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return -1

    max_area = -1
    max_idx = 0
    for i, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = box.cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            max_idx = i

    return max_idx


# ──────────────────────────────────────────────
# 팝업 3단계 자동 감지 (body metrics 기반)
# ──────────────────────────────────────────────

def extract_popup_stage_frames(
    video_path: str,
) -> dict[int, tuple[list, np.ndarray]]:
    """
    영상에서 팝업 3단계 대표 프레임을 body metrics로 자동 감지

    판별 기준:
      Stage 1 (Push)  : 앞 절반 중 shoulder_y 최대 → 가장 낮은 자세(엎드려 밀어올리는 순간)
      Stage 2 (Squat) : 전체 중 knee_angle 최소  → 무릎이 가장 많이 굽혀진 순간
      Stage 3 (Stand) : 뒤 절반 중 shoulder_y 최소 → 가장 일어선 자세

    Returns:
        {1: (kps_17, frame), 2: (kps_17, frame), 3: (kps_17, frame)}
        kps_17: [[x, y, conf], ...] 17개
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model = _get_model()

    # ── Phase 1: 전체 영상 스캔 (25프레임) → shoulder_y 궤적 ──
    scan_count = min(25, total_frames)
    scan_data = []
    for i in range(scan_count):
        frame_idx = int(total_frames * i / scan_count)
        d = _get_frame_data(cap, model, frame_idx)
        if d:
            scan_data.append(d)

    if len(scan_data) < 3:
        cap.release()
        raise ValueError(
            "영상에서 사람을 감지하지 못했습니다. 서퍼가 잘 보이는 영상을 사용해주세요."
        )

    # ── Phase 2: 팝업 구간 감지 ──
    shoulder_series = [
        (d["frame_idx"], d["shoulder_y"])
        for d in scan_data
        if d["shoulder_y"] < float("inf")
    ]
    popup_start, popup_end = _find_popup_window(shoulder_series, total_frames)

    # ── Phase 3: 팝업 구간 밀집 스캔 (15프레임) ──
    span = max(1, popup_end - popup_start)
    dense_count = min(15, span)
    dense_indices = sorted(set([
        int(popup_start + span * i / max(dense_count - 1, 1))
        for i in range(dense_count)
    ]))

    dense_data = []
    for frame_idx in dense_indices:
        d = _get_frame_data(cap, model, frame_idx)
        if d:
            dense_data.append(d)

    cap.release()

    # 밀집 스캔 실패 시 전체 스캔 결과로 대체
    if len(dense_data) < 3:
        dense_data = scan_data

    # ── Phase 4: 단계별 대표 프레임 선정 ──
    n = len(dense_data)

    # Stage 1: 앞 절반에서 shoulder_y 가장 큰 프레임 (가장 낮은 자세)
    first_half = dense_data[: max(1, int(n * 0.5))]
    stage1 = max(first_half, key=lambda d: d["shoulder_y"] if d["shoulder_y"] < float("inf") else 0)

    # Stage 2: 전체에서 knee_angle 가장 작은 프레임 (가장 쭈그린 자세)
    stage2 = min(dense_data, key=lambda d: d["knee_angle"])

    # Stage 3: 뒤 절반에서 shoulder_y 가장 작은 프레임 (가장 일어선 자세)
    second_half = dense_data[max(0, int(n * 0.5)):]
    stage3 = min(
        second_half,
        key=lambda d: d["shoulder_y"] if d["shoulder_y"] < float("inf") else 9999,
    )

    return {
        1: (stage1["kps"], stage1["frame"]),
        2: (stage2["kps"], stage2["frame"]),
        3: (stage3["kps"], stage3["frame"]),
    }


def _get_frame_data(cap, model, frame_idx: int) -> dict | None:
    """단일 프레임에서 키포인트 + 분류용 메트릭 추출"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None

    results = model(frame, verbose=False)
    if not results or results[0].keypoints is None:
        return None

    kps_data = results[0].keypoints
    if kps_data.xy is None or len(kps_data.xy) == 0:
        return None

    person_idx = _pick_main_person(results[0])
    if person_idx < 0:
        return None

    xy = kps_data.xy[person_idx].cpu().numpy()
    conf = kps_data.conf[person_idx].cpu().numpy()

    kps = [[float(xy[i][0]), float(xy[i][1]), float(conf[i])] for i in range(17)]

    # shoulder_y: 낮을수록 화면 위 = 일어선 상태
    sh_ys = [float(xy[j][1]) for j in [5, 6] if conf[j] > 0.3]
    shoulder_y = sum(sh_ys) / len(sh_ys) if sh_ys else float("inf")

    # knee_angle: 낮을수록 더 쭈그린 자세
    knee_angle = _compute_knee_angle(xy, conf)

    return {
        "frame_idx": frame_idx,
        "kps": kps,
        "frame": frame.copy(),
        "shoulder_y": shoulder_y,
        "knee_angle": knee_angle,
    }


def _compute_knee_angle(xy: np.ndarray, conf: np.ndarray) -> float:
    """무릎 각도 계산 (hip-knee-ankle 기준)"""

    def _angle3(a, b, c):
        ba = a - b
        bc = c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return math.degrees(math.acos(float(np.clip(cos_a, -1.0, 1.0))))

    angles = []
    # 왼쪽: hip(11)-knee(13)-ankle(15)
    if conf[11] > 0.3 and conf[13] > 0.3 and conf[15] > 0.3:
        angles.append(_angle3(xy[11], xy[13], xy[15]))
    # 오른쪽: hip(12)-knee(14)-ankle(16)
    if conf[12] > 0.3 and conf[14] > 0.3 and conf[16] > 0.3:
        angles.append(_angle3(xy[12], xy[14], xy[16]))

    return sum(angles) / len(angles) if angles else 180.0


def _find_popup_window(
    shoulder_series: list[tuple[int, float]],
    total_frames: int,
) -> tuple[int, int]:
    """
    어깨 y좌표 시계열에서 팝업 구간 감지

    팝업 = shoulder_y가 크게 감소하는 구간 (사람이 일어나는 중)
    stance 영상 = shoulder_y가 전체적으로 낮고 안정적 (이미 서 있음)

    Args:
        shoulder_series: [(frame_idx, shoulder_y), ...]
    Returns:
        (start_frame, end_frame)
    """
    if len(shoulder_series) < 3:
        return 0, total_frames - 1

    ys = [y for _, y in shoulder_series]
    total_variation = max(ys) - min(ys)

    # 변화량이 거의 없으면 전체 영상 사용
    if total_variation < 30:
        return shoulder_series[0][0], shoulder_series[-1][0]

    best_score = 0.0
    best_start = shoulder_series[0][0]
    best_end = shoulder_series[-1][0]

    for i in range(len(shoulder_series)):
        for j in range(i + 2, len(shoulder_series)):
            drop = shoulder_series[i][1] - shoulder_series[j][1]  # y 감소 = 올라감
            if drop < 20:  # 최소 20px 이상 하강해야 팝업으로 인정
                continue

            span_frames = shoulder_series[j][0] - shoulder_series[i][0]
            span_ratio = span_frames / total_frames

            # 너무 짧거나 너무 긴 구간 제외
            if span_ratio < 0.05 or span_ratio > 0.85:
                continue

            # 점수: 하강량이 크고 적당한 길이(30~50%)를 선호
            score = drop * (1.0 - abs(span_ratio - 0.35))

            if score > best_score:
                best_score = score
                best_start = shoulder_series[i][0]
                best_end = shoulder_series[j][0]

    return best_start, best_end
