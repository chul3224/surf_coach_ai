"""
레퍼런스 포즈 DB 빌드 스크립트

data/reference/ 폴더의 사진들로부터
동작별 평균 키포인트 벡터를 계산해 data/reference_poses.json 에 저장한다.

사용법:
    python build_reference_db.py          # 최초 빌드
    python build_reference_db.py --force  # 강제 재빌드
"""

import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(__file__))

from backend.analysis.reference_matcher import build_reference_db, get_reference_info

if __name__ == "__main__":
    force = "--force" in sys.argv
    db = build_reference_db(force=force)

    print("\n=== 레퍼런스 DB 현황 ===")
    for label, data in db.items():
        print(f"  {label:20s}: {data['sample_count']}장 유효 / {data['image_count']}장 전체")
    print("========================\n")
