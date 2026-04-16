# 🏄 SurfCoach AI

> AI가 서핑 자세를 분석하고, 전문가 수준의 피드백을 제공하는 코칭 앱
> 강사 5년 + 구조대 경력 기반의 도메인 지식으로 설계된 자세 분석 기준

---

## 현재 개발 상태 (2025-04-16 기준)

### 완료된 작업

#### 백엔드 (FastAPI)
- 영상 업로드 → 자세 분석 → LLM 피드백 → 결과 반환 API
- 분석 기록 저장 및 히스토리 조회 API
- SQLite DB 연동

#### LLM 추상화 레이어
- Claude / GPT-4o / Gemini / Gemma4 모델 교체 가능 구조
- 환경변수 `LLM_PROVIDER` 하나로 모델 전환
- 현재 테스트: Gemini 2.5 Flash Lite 사용 중

#### 자세 분석 엔진
- YOLOv8-pose로 17개 관절 키포인트 추출
- 팝업(Pop-up) 3단계 세부 분석
  - 1단계 푸쉬: 손 위치(갈비뼈 옆), 팔 완전히 펴기, 체중 뒤로 이동
  - 2단계 발 끌어오기: 쭈그리기 자세, 시선 방향(바닥 보면 빠짐)
  - 3단계 일어서기: 시선 처리, 무릎 각도, 상체 기울기
- 스탠스(Stance) 분석: 발 간격, 무게중심, 좌우 균형
- 패들링(Paddling) 분석: 어깨 대칭, 팔꿈치 높이, 머리 자세

#### 시각화
- 관절 점수별 색상 오버레이 이미지 자동 생성
  - 🟢 70~100점: 초록
  - 🟠 50~70점: 주황
  - 🔴 0~50점: 빨강
- 점수 패널 (우상단) 표시

---

## 내일 작업 예정 (3세션)

### 팝업 분석 정확도 개선
- 팝업 구간 직접 지정 파라미터 추가 (`start_sec`, `end_sec`)
  - 현재: 자동 감지 → 엉뚱한 stance 장면 분석되는 문제 있음
  - 개선: 사용자가 팝업 구간을 직접 지정
- 시선 감지 민감도 추가 조정

### React Native 프론트엔드 시작
- 영상 업로드 화면
- 분석 결과 화면 (오버레이 이미지 + 단계별 피드백)
- 히스토리 화면

---

## 기술 스택

| 파트 | 기술 |
|------|------|
| 백엔드 | FastAPI (Python) |
| 자세 감지 | YOLOv8-pose |
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

# 3) 서버 실행
uvicorn backend.main:app --reload

# 4) API 문서 확인
# http://localhost:8000/docs
```
