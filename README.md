# 효도 용돈박스 상품 사진 배경/조명 정제 PoC

AIFFEL 모듈 PoC 과제 — 자기 도메인 병목을 멀티모달 AI(YOLOv8-seg + Diffusion inpainting)로 개선.

## 제출 문서
- **문제 정의서**: [`docs/PROBLEM_DEFINITION.md`](docs/PROBLEM_DEFINITION.md) (도메인, 현재의 문제, 개선 가설, 대상 사용자, 성공 기준)
- **PoC 코드**: [`src/segment.py`](src/segment.py), [`src/inpaint.py`](src/inpaint.py), [`app.py`](app.py) — 모델 선정 근거는 [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md) 참고
- **개선 효과 검증 결과**: [`docs/RESULTS.md`](docs/RESULTS.md) (기존 수작업과의 비교, 실패 사례 포함)
- **README**(이 문서): 실행 방법 + 아래 [실행 결과](#실행-결과)의 시연 자료

## 도메인 및 문제
1688에서 옥 지압봉을 소싱해 "효도 용돈박스" 형태로 리패키징하여 쿠팡 등에 판매 예정. 스튜디오 장비 없이 집에서 직접 촬영한 상품 사진은 배경이 지저분하거나 조명이 고르지 않아 그대로 상세페이지에 쓰기 어려움. 자세한 내용은 [`docs/PROBLEM_DEFINITION.md`](docs/PROBLEM_DEFINITION.md) 참고.

## 파이프라인
```
촬영 이미지 → YOLOv8-seg(상품 영역 분할) → Diffusion inpainting(배경 재생성) → 합성 → 정제된 상세페이지용 이미지
```

모델 선정 근거(상품 영역 분할, 배경 재생성 각 단계의 대안 비교)는 [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md)에 정리했다.

## 환경 준비
```bash
# Python 3.12, uv 사용
uv venv
uv pip install -r requirements.txt
```
- YOLOv8-seg 가중치(`weights/yolov8n-seg.pt`)는 `ultralytics`가 최초 실행 시 자동 다운로드합니다.
- `runwayml/stable-diffusion-inpainting` 모델은 최초 실행 시 Hugging Face 캐시에 자동 다운로드됩니다(fp16, 약 4GB).
- GPU(CUDA) 권장 — 이 프로젝트는 RTX 3060(12GB)에서 검증했습니다. GPU가 없으면 CPU로도 동작하지만 매우 느립니다.

## 사용법
```bash
# 1) 분할: 마스크 오버레이 확인용 (data/output/masks/에 저장)
python src/segment.py

# 2) 배경 정제: data/raw/의 전체 이미지를 배경만 diffusion inpainting (data/output/refined/에 저장)
python src/inpaint.py

# 3) Gradio 데모: 사진 한 장을 업로드해 바로 결과 확인
python app.py
```
`app.py` 실행 후 터미널에 뜨는 로컬 주소(기본 http://127.0.0.1:7860)로 접속하면 됩니다.

## 실행 결과

`python app.py` 실행 후 실제로 사진을 업로드해 나온 결과 (Gradio 데모):

![Gradio 데모 실행 결과](docs/screenshots/gradio_demo_result.jpg)

수작업(클립스튜디오) vs AI 결과 비교 등 더 많은 사례는 [`docs/RESULTS.md`](docs/RESULTS.md)에서 볼 수 있다:

![수작업 vs AI 3자 비교](docs/comparisons/toothpick_3way_compare.jpg)

- `data/output/masks/` — YOLOv8-seg 분할 결과 오버레이 (20장 전체)
- `data/output/refined/` — 배경 정제 최종 결과 (20장 전체)

## 개선 효과 검증 결과
20장 전체에 대한 성공/실패 분류, 원인 분석, 수작업 대비 처리 시간 비교는
[`docs/RESULTS.md`](docs/RESULTS.md)에 정리했다.
