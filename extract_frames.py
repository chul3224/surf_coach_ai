"""
서핑 영상 → 이미지 프레임 추출기

사용법:
    python extract_frames.py                          # 기본 (1초당 1장)
    python extract_frames.py --fps 2                  # 1초당 2장
    python extract_frames.py --every 30               # 30프레임마다 1장
    python extract_frames.py --max 50                 # 영상당 최대 50장

    # 초반 구간 집중 추출 (테이크오프 포착용)
    python extract_frames.py --first 10               # 처음 10초, 초당 10장
    python extract_frames.py --first 10 --dense 5    # 처음 10초, 초당 5장
    python extract_frames.py --start 3 --end 12      # 3~12초 구간, 초당 10장
"""

import cv2
import argparse
from pathlib import Path

# ── 설정 ──────────────────────────────────────────
VIDEO_DIR = Path(r"C:\Users\chul3\OneDrive\Desktop\우철\서핑영상 AI학습용")
OUTPUT_DIR = Path(r"C:\Users\chul3\github\surf_coach_ai\data\frames")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".MP4", ".MOV", ".AVI"}
# ──────────────────────────────────────────────────


def extract_frames(
    video_path: Path,
    output_dir: Path,
    every_n_frames: int = None,
    fps_target: float = 1.0,
    max_frames: int = None,
    start_sec: float = 0.0,
    end_sec: float = None,
):
    """단일 영상에서 프레임 추출"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [오류] 열 수 없음: {video_path.name}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps

    # 구간 프레임 번호 계산
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps) if end_sec else total_frames
    end_frame = min(end_frame, total_frames)
    span_sec = (end_frame - start_frame) / video_fps

    # 추출 간격 결정
    if every_n_frames:
        interval = every_n_frames
    else:
        interval = max(1, int(video_fps / fps_target))

    estimated = (end_frame - start_frame) // interval

    # 출력 폴더: data/frames/영상이름/
    video_out = output_dir / video_path.stem
    video_out.mkdir(parents=True, exist_ok=True)

    print(f"\n  영상: {video_path.name}")
    print(f"  전체: {total_frames}프레임 | {duration_sec:.1f}초 | {video_fps:.0f}fps")
    print(f"  구간: {start_sec:.1f}초 ~ {(end_frame/video_fps):.1f}초 ({span_sec:.1f}초)")
    print(f"  간격: {interval}프레임마다 1장 ({fps_target:.0f}fps) → 예상 {estimated}장")

    # 시작 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved = 0
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % interval == 0:
            # 파일명에 타임스탬프(초) 포함 → 나중에 라벨링 시 위치 파악 쉬움
            timestamp = frame_idx / video_fps
            filename = video_out / f"t{timestamp:07.3f}s_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

            if max_frames and saved >= max_frames:
                print(f"  최대 {max_frames}장 도달, 중단")
                break

        frame_idx += 1

    cap.release()
    print(f"  저장 완료: {saved}장 → {video_out}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="서핑 영상 프레임 추출기")
    parser.add_argument("--fps", type=float, default=1.0, help="초당 추출 장수 (기본: 1.0)")
    parser.add_argument("--every", type=int, default=None, help="N 프레임마다 1장 추출")
    parser.add_argument("--max", type=int, default=None, help="영상당 최대 추출 장수")
    parser.add_argument("--video", type=str, default=None, help="특정 영상 파일명만 추출")

    # 구간 지정 옵션
    parser.add_argument("--first", type=float, default=None,
                        help="처음 N초 구간만 촘촘하게 추출 (기본 10fps)")
    parser.add_argument("--dense", type=float, default=10.0,
                        help="--first 사용 시 초당 추출 장수 (기본: 10)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="추출 시작 시간(초)")
    parser.add_argument("--end", type=float, default=None,
                        help="추출 종료 시간(초)")

    args = parser.parse_args()

    # --first 옵션 처리
    if args.first is not None:
        args.start = 0.0
        args.end = args.first
        args.fps = args.dense

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 영상 파일 수집 (하위 폴더 포함)
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(VIDEO_DIR.rglob(f"*{ext}"))

    # 중복 제거
    seen = set()
    unique_videos = []
    for v in videos:
        if v.name not in seen:
            seen.add(v.name)
            unique_videos.append(v)
    videos = sorted(unique_videos)

    # 특정 영상만 추출
    if args.video:
        videos = [v for v in videos if args.video.lower() in v.name.lower()]

    if not videos:
        print("영상 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(videos)}개 영상 발견")
    print(f"출력 경로: {OUTPUT_DIR}")
    if args.first:
        print(f"모드: 처음 {args.first}초 구간, 초당 {args.fps:.0f}장 추출")
    elif args.end:
        print(f"모드: {args.start}초 ~ {args.end}초 구간, 초당 {args.fps:.0f}장 추출")
    print("=" * 50)

    total_saved = 0
    for video in videos:
        count = extract_frames(
            video_path=video,
            output_dir=OUTPUT_DIR,
            every_n_frames=args.every,
            fps_target=args.fps,
            max_frames=args.max,
            start_sec=args.start,
            end_sec=args.end,
        )
        total_saved += count

    print("\n" + "=" * 50)
    print(f"전체 완료: 총 {total_saved}장 추출")
    print(f"저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
