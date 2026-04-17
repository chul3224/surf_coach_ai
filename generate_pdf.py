"""
SurfCoach AI 프로젝트 제안서 PDF 생성기
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import os

# ── 한글 폰트 등록 ──
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/malgunbd.ttf"

pdfmetrics.registerFont(TTFont("Malgun", FONT_PATH))
pdfmetrics.registerFont(TTFont("MalgunBold", FONT_BOLD_PATH))

# ── 컬러 팔레트 (참고 기획서 스타일) ──
NAVY      = colors.HexColor("#1a2e4a")
TEAL      = colors.HexColor("#0e7490")
LIGHT_BG  = colors.HexColor("#f0f9ff")
GRAY_BG   = colors.HexColor("#f8f9fa")
BORDER    = colors.HexColor("#e2e8f0")
TEXT      = colors.HexColor("#1e293b")
MUTED     = colors.HexColor("#64748b")
GREEN     = colors.HexColor("#16a34a")
ORANGE    = colors.HexColor("#d97706")
WHITE     = colors.white

W, H = A4
MARGIN = 18 * mm

# ── 스타일 정의 ──
def make_styles():
    s = {}

    s["title"] = ParagraphStyle(
        "title", fontName="MalgunBold", fontSize=26,
        textColor=WHITE, alignment=TA_CENTER,
        spaceAfter=2, leading=34,
    )
    s["subtitle"] = ParagraphStyle(
        "subtitle", fontName="Malgun", fontSize=13,
        textColor=colors.HexColor("#bae6fd"), alignment=TA_CENTER,
        spaceAfter=2, leading=18,
    )
    s["tagline"] = ParagraphStyle(
        "tagline", fontName="Malgun", fontSize=10,
        textColor=colors.HexColor("#7dd3fc"), alignment=TA_CENTER,
        spaceAfter=0, leading=14,
    )
    s["section"] = ParagraphStyle(
        "section", fontName="MalgunBold", fontSize=13,
        textColor=WHITE, alignment=TA_LEFT,
        spaceBefore=0, spaceAfter=0, leading=18,
        leftIndent=0,
    )
    s["body"] = ParagraphStyle(
        "body", fontName="Malgun", fontSize=9.5,
        textColor=TEXT, leading=15, spaceAfter=3,
    )
    s["body_bold"] = ParagraphStyle(
        "body_bold", fontName="MalgunBold", fontSize=9.5,
        textColor=TEXT, leading=15, spaceAfter=3,
    )
    s["bullet"] = ParagraphStyle(
        "bullet", fontName="Malgun", fontSize=9,
        textColor=TEXT, leading=14, spaceAfter=2,
        leftIndent=8, firstLineIndent=-8,
    )
    s["feature_title"] = ParagraphStyle(
        "feature_title", fontName="MalgunBold", fontSize=10,
        textColor=NAVY, leading=14, spaceAfter=3,
    )
    s["scenario"] = ParagraphStyle(
        "scenario", fontName="Malgun", fontSize=9,
        textColor=colors.HexColor("#0c4a6e"), leading=14,
        leftIndent=6,
    )
    s["phase_title"] = ParagraphStyle(
        "phase_title", fontName="MalgunBold", fontSize=10,
        textColor=TEAL, leading=14, spaceAfter=4,
    )
    s["code"] = ParagraphStyle(
        "code", fontName="Malgun", fontSize=8,
        textColor=colors.HexColor("#334155"), leading=13,
        leftIndent=6, backColor=GRAY_BG,
    )
    s["footer"] = ParagraphStyle(
        "footer", fontName="Malgun", fontSize=8,
        textColor=MUTED, alignment=TA_CENTER, leading=12,
    )
    s["tbl_header"] = ParagraphStyle(
        "tbl_header", fontName="MalgunBold", fontSize=9,
        textColor=WHITE, leading=13, alignment=TA_CENTER,
    )
    s["tbl_cell"] = ParagraphStyle(
        "tbl_cell", fontName="Malgun", fontSize=8.5,
        textColor=TEXT, leading=13,
    )
    s["tbl_cell_bold"] = ParagraphStyle(
        "tbl_cell_bold", fontName="MalgunBold", fontSize=8.5,
        textColor=TEXT, leading=13,
    )
    return s

S = make_styles()

# ── 헬퍼 ──
def section_header(num_label, title_text):
    """섹션 헤더 블록 (네이비 배경)"""
    label = Paragraph(f"{num_label}  {title_text}", S["section"])
    tbl = Table([[label]], colWidths=[W - 2 * MARGIN])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    return tbl

def feature_card(title, bullets):
    inner = [Paragraph(title, S["feature_title"])]
    for b in bullets:
        inner.append(Paragraph(f"• {b}", S["bullet"]))
    tbl = Table([[inner]], colWidths=[W - 2 * MARGIN])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BG),
        ("BOX",           (0, 0), (-1, -1), 0.8, TEAL),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    return tbl

def scenario_box(lines):
    inner = [Paragraph("📋  서비스 시나리오", S["body_bold"])]
    for line in lines:
        inner.append(Paragraph(line, S["scenario"]))
    tbl = Table([[inner]], colWidths=[W - 2 * MARGIN])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#e0f2fe")),
        ("BOX",           (0, 0), (-1, -1), 1, TEAL),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    return tbl

def phase_card(label, title, status, bullets):
    status_color = {"✅": GREEN, "🔄": ORANGE, "🔲": MUTED}
    sc = status_color.get(status[:1], MUTED)
    header = Paragraph(f"{label}  {title}  {status}", S["phase_title"])
    rows = [[header]]
    for b in bullets:
        rows.append([Paragraph(f"• {b}", S["bullet"])])
    inner_tbl = Table(rows, colWidths=[W - 2 * MARGIN - 24])
    inner_tbl.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))
    tbl = Table([[inner_tbl]], colWidths=[W - 2 * MARGIN])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), GRAY_BG),
        ("LINEAFTER",     (0, 0), (0, -1), 3, sc),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    return tbl

def make_table(header_row, data_rows, col_widths, center_cols=None):
    center_cols = center_cols or []
    all_rows = [[Paragraph(h, S["tbl_header"]) for h in header_row]]
    for row in data_rows:
        all_rows.append([Paragraph(str(c), S["tbl_cell"]) for c in row])

    tbl = Table(all_rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("BACKGROUND",    (0, 1), (-1, -1), WHITE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GRAY_BG]),
        ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    for col in center_cols:
        style.append(("ALIGNMENT", (col, 0), (col, -1), "CENTER"))
    tbl.setStyle(TableStyle(style))
    return tbl

def asis_tobe_table(rows):
    """AS-IS / TO-BE 2열 표"""
    header = [
        Paragraph("현장의 문제점  (AS-IS)", S["tbl_header"]),
        Paragraph("SurfCoach AI의 해결책  (TO-BE)", S["tbl_header"]),
    ]
    data = [header]
    for left, right in rows:
        data.append([
            Paragraph(left,  S["tbl_cell"]),
            Paragraph(right, S["tbl_cell"]),
        ])
    cw = [(W - 2 * MARGIN) / 2] * 2
    tbl = Table(data, colWidths=cw, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, 0), colors.HexColor("#b91c1c")),
        ("BACKGROUND",    (1, 0), (1, 0), TEAL),
        ("BACKGROUND",    (0, 1), (0, -1), colors.HexColor("#fff1f2")),
        ("BACKGROUND",    (1, 1), (1, -1), LIGHT_BG),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return tbl

# ── PDF 빌드 ──
def build_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )

    story = []
    sp = lambda n=4: Spacer(1, n * mm)
    hr = lambda: HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=3)

    # ── 표지 ──
    cover = Table(
        [[
            Paragraph("SurfCoach AI", S["title"]),
            Paragraph("AI 기반 서핑 자세 분석 코칭 솔루션", S["subtitle"]),
            Paragraph("Project Proposal · 프로젝트 제안서", S["tagline"]),
        ]],
        colWidths=[W - 2 * MARGIN],
    )
    cover.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    story.append(cover)
    story.append(sp(6))

    # ── 00. 프로젝트 개요 ──
    story.append(section_header("00", "프로젝트 개요"))
    story.append(sp(2))
    story.append(make_table(
        ["항목", "내용"],
        [
            ["시스템 명칭", "SurfCoach AI"],
            ["한 줄 정의", "서핑 영상을 업로드하면 AI가 자세를 분석하고\n전문 강사 수준의 피드백을 제공하는 코칭 앱"],
            ["타겟 유저", "입문~중급 서퍼 (독학러, 레슨비 부담층)"],
            ["기획자 배경", "서핑 강사 5년 + 해양 구조대 경력 → 현장 도메인 지식 기반 설계"],
            ["GitHub", "https://github.com/chul3224/surf_coach_ai"],
        ],
        [50 * mm, W - 2 * MARGIN - 50 * mm],
    ))
    story.append(sp(5))

    # ── 01. 기획 배경 ──
    story.append(section_header("01", "기획 배경 및 문제 해결 방향"))
    story.append(sp(2))
    story.append(asis_tobe_table([
        [
            "레슨비 부담: 1회 레슨 5~10만원 수준,\n반복 교정은 비용이 기하급수적으로 증가",
            "무제한 자세 교정: 영상 한 편으로 즉시 분석\n→ 레슨 없이 스스로 교정 가능"
        ],
        [
            "즉각 피드백 불가: 파도에서 나온 후\n기억에 의존해 자세를 복기해야 함",
            "영상 기반 객관적 분석: 느낌이 아닌\n수치(각도·점수)로 자세를 진단"
        ],
        [
            "강사 기준 불투명: 강사마다 교정 기준이\n달라 학습자가 혼란",
            "도메인 지식 내재화: 강사 경력 5년의\n체크포인트를 AI 분석 기준으로 코드화"
        ],
        [
            "위험 자세 인식 부족: 낙수로 이어지는\n자세를 스스로 파악하기 어려움",
            "낙수 위험 자세 감지: 구조대 경력 기반\n안전 기준 추가 (시선 하향 등)"
        ],
    ]))
    story.append(sp(3))
    story.append(scenario_box([
        "1. 서퍼가 스마트폰으로 서핑 영상 촬영",
        "2. 앱에 영상 업로드 + 동작 선택 (테이크오프 / 스탠스 / 패들링)",
        "3. AI가 17개 관절 키포인트 추출 → 단계별 자세 분석",
        "4. 점수(0~100) + 관절 오버레이 이미지 + LLM 교정 피드백 반환",
        "5. 히스토리에서 날짜별 성장 추이 확인",
    ]))
    story.append(sp(5))

    # ── 02. 주요 기능 ──
    story.append(section_header("02", "주요 기능"))
    story.append(sp(2))

    cards = [
        ("테이크오프 Take-off — 3단계 자동 분석", [
            "Body metrics 기반 3단계 자동 감지 (단순 시간 분할 방식 탈피)",
            "1단계 푸쉬: 팔 펴짐 각도, 손 위치(갈비뼈 옆)",
            "2단계 발 끌어오기: 무릎 각도(90~130°), 시선 하향 감지 ← 낙수 핵심 원인",
            "3단계 일어서기: 시선 방향, 무릎 안정 각도(90~120°), 상체 기울기",
        ]),
        ("스탠스 Stance — 파도 위 자세 분석", [
            "발 간격 (어깨너비 기준 1.0~1.5배)",
            "무게중심 분배 (앞발 60% / 뒷발 40%)",
            "좌우 무릎 각도 균형",
        ]),
        ("패들링 Paddling — 추진력 자세 분석", [
            "팔 뻗음 정도 (충분히 앞으로 뻗어야 팔꿈치 수직 입수 가능)",
            "몸통 좌우 대칭 (보드 중심 유지 여부)",
            "스트로크 좌우 균형 / 머리 위치",
        ]),
        ("자세 오버레이 시각화", [
            "17개 관절 스켈레톤 + 점수별 색상 표시",
            "초록: 70점 이상 (Good) / 주황: 50~70 (Check) / 빨강: 50 미만 (Fix)",
        ]),
        ("LLM 교정 피드백", [
            "분석 수치 → 자연어 교정 가이드 자동 생성",
            "Claude / GPT-4o / Gemini / Gemma4 교체 가능 구조 (현재: Gemini 2.5 Flash)",
        ]),
    ]
    for title, bullets in cards:
        story.append(feature_card(title, bullets))
        story.append(sp(2))
    story.append(sp(3))

    # ── 03. 기술 스택 ──
    story.append(section_header("03", "기술 스택"))
    story.append(sp(2))
    story.append(Paragraph("모델 구성", S["body_bold"]))
    story.append(sp(1))
    story.append(make_table(
        ["구분", "기술", "역할"],
        [
            ["자세 감지", "YOLOv8-pose", "17개 관절 키포인트 추출"],
            ["피드백 생성 (운영)", "Gemini 2.5 Flash", "자연어 교정 문장 생성"],
            ["피드백 생성 (목표)", "Gemma4 로컬", "온디바이스 추론, 비용 절감"],
            ["피드백 생성 (검증)", "Claude / GPT-4o", "베이스라인 품질 비교"],
        ],
        [44 * mm, 54 * mm, W - 2 * MARGIN - 98 * mm],
    ))
    story.append(sp(3))
    story.append(Paragraph("전체 기술 스택", S["body_bold"]))
    story.append(sp(1))
    story.append(make_table(
        ["파트", "기술", "완성 여부"],
        [
            ["백엔드", "FastAPI (Python)", "✅ 완성"],
            ["자세 분석 엔진", "YOLOv8-pose + NumPy", "✅ 완성"],
            ["3단계 자동 감지", "Body metrics (어깨 높이 / 무릎 각도)", "✅ 완성"],
            ["LLM 추상화", "Factory 패턴 (4개 모델 교체 가능)", "✅ 완성"],
            ["시각화", "OpenCV 오버레이", "✅ 완성"],
            ["데이터베이스", "SQLite", "✅ 운영 중"],
            ["레퍼런스 포즈 매칭", "YOLO + 코사인 유사도", "🔄 구현 예정"],
            ["프론트엔드", "React Native", "🔲 예정"],
            ["음성 피드백", "gTTS / Whisper", "🔲 고도화 단계"],
            ["클라우드 배포", "AWS / GCP", "🔲 예정"],
        ],
        [44 * mm, 80 * mm, W - 2 * MARGIN - 124 * mm],
        center_cols=[2],
    ))
    story.append(sp(5))

    # ── 04. 개발 파이프라인 ──
    story.append(section_header("04", "개발 파이프라인"))
    story.append(sp(2))

    phases = [
        ("Phase 1", "핵심 백엔드 완성", "✅", [
            "YOLOv8-pose 서핑 영상 자세 분석 엔진 구축",
            "테이크오프 3단계 body metrics 자동 감지 알고리즘",
            "스탠스 / 패들링 도메인 지식 기반 체크포인트 설계",
            "FastAPI 서버 (영상 업로드 → 분석 → 결과 반환)",
            "LLM 추상화 레이어 (Claude / GPT-4o / Gemini / Gemma4)",
            "관절 점수별 색상 오버레이 이미지 생성",
        ]),
        ("Phase 2", "레퍼런스 포즈 매칭", "🔄", [
            "이상적 자세 레퍼런스 사진 라벨링 (테이크오프 Push/Squat/Standup, 스탠스, 패들링)",
            "YOLO로 레퍼런스 키포인트 추출 및 저장",
            "사용자 프레임 vs 레퍼런스 코사인 유사도 비교",
            "동작 자동 분류 정확도 향상 (스탠스 vs 테이크오프 구분 문제 해결)",
        ]),
        ("Phase 3", "프론트엔드 (React Native)", "🔲", [
            "영상 업로드 화면",
            "분석 결과 화면 (점수 카드 + 오버레이 이미지)",
            "히스토리 화면 (날짜별 기록, 성장 그래프)",
        ]),
        ("Phase 4", "고도화 및 출시", "🔲", [
            "음성 피드백 (TTS)",
            "실제 서퍼 베타 테스트 및 피드백 반영",
            "클라우드 배포 (AWS / GCP)",
            "정식 출시",
        ]),
    ]
    for label, title, status, bullets in phases:
        story.append(phase_card(label, title, status, bullets))
        story.append(sp(2))
    story.append(sp(3))

    # ── 05. 현재 진행 상황 ──
    story.append(section_header("05", "현재 진행 상황"))
    story.append(sp(2))
    current = [
        "백엔드 API 완성 (POST /analyze, GET /history, GET /history/{id})",
        "테이크오프 3단계 분석 운영 중 (Gemini 2.5 Flash 피드백 생성)",
        "서핑 영상 11개에서 프레임 330장 추출 완료 (data/frames/)",
        "레퍼런스 포즈 매칭 시스템 구현 완료 (reference_matcher.py)",
        "주 피사체 선택 알고리즘 개선 (중앙 근접도 + 면적 복합 기준)",
        "서버 테스트 완료 — takeoff / stance / paddling 분석 정상 동작",
    ]
    inner = []
    for item in current:
        inner.append(Paragraph(f"• {item}", S["bullet"]))
    box = Table([[inner]], colWidths=[W - 2 * MARGIN])
    box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BG),
        ("BOX",           (0, 0), (-1, -1), 0.8, TEAL),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
    ]))
    story.append(box)
    story.append(sp(5))

    # ── 06. 향후 과제 및 기대 효과 ──
    story.append(section_header("06", "향후 과제 및 기대 효과"))
    story.append(sp(2))
    story.append(make_table(
        ["구분", "내용"],
        [
            ["레퍼런스 데이터 확보", "동작별 이상 자세 레퍼런스 사진 수집 및 YOLO 키포인트 추출"],
            ["분석 정확도 검증", "실제 서핑 영상으로 단계 감지 정확도 측정 및 튜닝"],
            ["LLM 품질 비교", "Claude / GPT-4o / Gemini / Gemma4 피드백 품질 A/B 테스트"],
            ["배포 모델 확정", "비용·속도·품질 균형점 → Gemma4 로컬 배포 목표"],
            ["기대 효과", "레슨비 부담 없이 언제든 자세 교정 가능, 서핑 입문 장벽 낮춤"],
        ],
        [50 * mm, W - 2 * MARGIN - 50 * mm],
    ))
    story.append(sp(5))

    # ── 07. 경쟁 우위 ──
    story.append(section_header("07", "경쟁 우위"))
    story.append(sp(2))
    story.append(make_table(
        ["일반 AI 서비스", "SurfCoach AI"],
        [
            ["자세 기준 불명확", "강사 5년 경험 기반 체크포인트 직접 코드화"],
            ["위험 판단 불가", "구조대 경력 기반 낙수 위험 자세(시선 하향 등) 감지"],
            ["서핑 특화 없음", "테이크오프 3단계 / 스탠스 / 패들링 전문 분석"],
            ["단일 LLM 종속", "Claude / GPT-4o / Gemini / Gemma4 교체 가능 구조"],
            ["주관적 피드백", "각도·비율 수치 기반 객관적 점수 + 교정 가이드"],
        ],
        [(W - 2 * MARGIN) / 2] * 2,
    ))
    story.append(sp(6))

    # ── 푸터 ──
    story.append(hr())
    story.append(Paragraph("추후 진행에 따라 내용이 변경될 수 있습니다.", S["footer"]))

    doc.build(story)
    print(f"PDF 생성 완료: {output_path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "SurfCoach_AI_프로젝트_제안서.pdf")
    build_pdf(out)
