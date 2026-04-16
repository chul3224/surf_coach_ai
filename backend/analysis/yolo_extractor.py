"""
YOLOv8-pose 키포인트 추출기

업로드된 영상에서 대표 프레임을 뽑고
YOLOv8-pose 로 17개 관절 좌표를 추출한다.

반환 형식: [[x, y, confidence], ...] 17개
"""

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
