# 🏄 SurfCoach AI

> AI가 서핑 자세를 분석하고, 전문가 수준의 피드백을 제공하는 코칭 앱
> 강사 5년 + 구조대 경력 기반의 도메인 지식으로 설계된 자세 분석 기준

---

## 현재 개발 상태 (2026-04-19 기준)

### 완료된 작업

#### 백엔드 (FastAPI)
- 영상 업로드 → 자세 분석 → LLM 피드백 → 결과 반환 API
- 분석 기록 저장 및 히스토리 조회 API
- SQLite DB 연동

#### LLM 추상화 레이어
- Claude / GPT-4o / Gemini / Gemma4 모델 교체 가능 구조
- 환경변수 `LLM_PROVIDER` 하나로 모델 전환
- 현재: Gemini 2.5 Flash 사용 중

#### 자세 분석 엔진
- **YOLO26s-pose**로 17개 관절 키포인트 추출 (YOLOv8n → YOLO26s 업그레이드)
- 테이크오프(Takeoff) 3단계 세부 분석
  - 1단계 푸쉬(Push): 손 위치(갈비뼈 옆), 팔 완전히 펴기, 체중 뒤로 이동
  - 2단계 발 끌어오기(Pull & Squat): 쭈그리기 자세, 시선 방향
  - 3단계 일어서기(Stand Up): 시선 처리, 무릎 각도, 상체 기울기
- 스탠스(Stance) 분석: 발 간격, 무게중심, 좌우 균형
- 패들링(Paddling) 분석: 어깨 대칭, 팔꿈치 높이, 머리 자세
- body metrics 기반 테이크오프 3단계 자동 감지 (어깨 y좌표 + 무릎 각도)

#### 레퍼런스 포즈 매칭 시스템
- 동작별 레퍼런스 사진 → 평균 키포인트 벡터 저장 (`data/reference_poses.json`)
- 코사인 유사도로 사용자 프레임 vs 레퍼런스 비교
- 감지율: stance 13/15, takeoff_push 9/21, takeoff_standup 7/11
- 빌드: `python build_reference_db.py --force`

#### 주 피사체 선택
- 다중 서퍼 영상에서 중앙 근접도 60% + 면적 40% 복합 기준으로 주 피사체 선택

#### 시각화
- 관절 점수별 색상 오버레이 이미지 자동 생성
  - 70~100점: 초록
  - 50~70점: 주황
  - 0~50점: 빨강
- 테이크오프 3단계 각각 별도 오버레이 이미지 저장

---

## 다음 작업 예정

### React Native 프론트엔드
- 홈 화면 — 동작 선택 (테이크오프 / 스탠스 / 패들링)
- 영상 업로드 화면
- 분석 결과 화면 (점수 카드 + 오버레이 이미지 + 단계별 피드백)
- 히스토리 화면 (날짜별 기록, 성장 그래프)

### 레퍼런스 보강
- paddling 레퍼런스 감지율 개선 (현재 1/34 — 지상 촬영 사진 추가 필요)

---

## 기술 스택

| 파트 | 기술 |
|---|---|
| 백엔드 | FastAPI (Python) |
| 자세 감지 | YOLO26s-pose |
| 피드백 생성 | Gemini 2.5 Flash (개발) → Gemma4 로컬 (배포 목표) |
| DB | SQLite → PostgreSQL (배포 시) |
| 영상 처리 | OpenCV |
| 프론트엔드 | React Native (예정) |

---

## 로컬 실행 방법

```bash
# 1) 환경변수 설정
cp .env.example .env
# .env 파일에 GOOGLE_API_KEY 입력, LLM_PROVIDER=gemini

# 2) 패키지 설치
pip install -r requirements.txt

# 3) 레퍼런스 DB 빌드 (최초 1회)
python build_reference_db.py

# 4) 서버 실행
uvicorn backend.main:app --reload

# 5) API 문서 확인
# http://localhost:8000/docs
```
