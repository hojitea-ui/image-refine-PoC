# 효도 용돈박스 상품 사진 배경/조명 정제 PoC

AIFFEL 모듈 PoC 과제 — 자기 도메인 병목을 멀티모달 AI(YOLOv8-seg + Diffusion inpainting)로 개선.

## 제출 문서
- **문제 정의서**: [`docs/PROBLEM_DEFINITION.md`](docs/PROBLEM_DEFINITION.md) (도메인, 현재의 문제, 개선 가설, 대상 사용자, 성공 기준)
- **PoC 코드**: [`src/segment.py`](src/segment.py), [`src/inpaint.py`](src/inpaint.py), [`app.py`](app.py) — 모델 선정 근거는 아래 [모델 선정 근거](#모델-선정-근거) 참고
- **개선 효과 검증 결과**: [`docs/RESULTS.md`](docs/RESULTS.md) (기존 수작업과의 비교, 실패 사례 포함)
- **README**(이 문서): 실행 방법 + 아래 [실행 결과](#실행-결과)의 시연 자료

## 도메인 및 문제
1688에서 옥 지압봉을 소싱해 "효도 용돈박스" 형태로 리패키징하여 쿠팡 등에 판매 예정. 스튜디오 장비 없이 집에서 직접 촬영한 상품 사진은 배경이 지저분하거나 조명이 고르지 않아 그대로 상세페이지에 쓰기 어려움. 자세한 내용은 [`docs/PROBLEM_DEFINITION.md`](docs/PROBLEM_DEFINITION.md) 참고.

## 파이프라인
```
촬영 이미지 → YOLOv8-seg(상품 영역 분할) → Diffusion inpainting(배경 재생성) → 합성 → 정제된 상세페이지용 이미지
```

## 모델 선정 근거

상품 영역 분할과 배경 재생성 두 단계 각각에서 대안을 검토하고 골랐다. (초기 기획 단계에서 "워터마크
제거" 문제로 논의했던 비교를 현재 "배경/조명 정제" 목표에 맞게 다시 정리한 것 — 배경 피벗 경위는
`docs/PROBLEM_DEFINITION.md` 참고.)

### ① 상품 영역 분할 — YOLOv8-seg vs SAM vs rembg

| 모델 | 방식 | 장점 | 단점 |
|---|---|---|---|
| **YOLOv8-seg (채택)** | 클래스 기반 자동 탐지+분할 | 프롬프트 설계 불필요, 가볍고 빠름, 모듈에서 다룬 YOLO 계열 | 학습되지 않은 상품 카테고리는 인식 못할 수 있음 |
| SAM | 포인트/박스 프롬프트로 분할 | 범용성·정확도 최고 | 프롬프트 로직을 별도로 설계해야 하고 무겁고 느림 |
| rembg (U2-Net) | 배경 제거 전용 경량 모델 | 세팅이 가장 간단, 단일 객체 사진에 결과가 깔끔한 경우 많음 | 모듈에서 다룬 계열은 아님 |

**선정 이유**: YOLOv8-seg는 학습된 클래스 기반으로 곧바로 분할이 가능해 구현 속도가 빠르고, 모듈에서
다룬 YOLO 계열에 속해 적용 근거가 명확하다. 실제로 상품이 COCO 80종에 없는 경우(효도박스, 스피커 등)
탐지에 실패하는 한계가 발견되어, 면적 비율 필터링 + GrabCut 폴백(`get_product_mask()`,
`grabcut_fallback()` in `src/inpaint.py`)을 추가로 구현했다.

### ② 배경 재생성 — Diffusion inpainting vs LaMa vs OCR+inpainting

| 모델 | 방식 | 장점 | 단점 |
|---|---|---|---|
| **Diffusion inpainting (채택)** | 마스크 영역을 생성형으로 채움 | 넓은 영역도 자연스러운 텍스처로 복원, 모듈에서 다룬 생성형 계열 | 무겁고, 가끔 배경과 무관한 내용을 "환각"으로 채워 넣음 |
| LaMa | inpainting 특화 경량 모델 | 국소적인 영역(워터마크 등)을 지우는 데 diffusion보다 안정적이고 빠름 | 배경 전체를 새로운 장면으로 "생성"하는 용도엔 덜 적합, 모듈 밖 모델 |
| OCR 탐지 + Diffusion inpainting | 텍스트 영역 자동 탐지 후 inpainting | 마스크를 수동으로 안 그려도 됨 | 모델이 두 개 엮여 구현 복잡도 증가, 배경 전체 재생성이라는 이번 문제와는 결이 다름 |

**선정 이유**: Diffusion inpainting은 텍스처 복원이 자연스러워 "상세페이지에 바로 쓸 수 있는 수준"이라는
목표에 부합한다고 판단해 채택했다. 다만 20장 실제 테스트 결과 이 장점이 곧 단점으로도 나타났다 —
inpainting 대상 영역이 넓을수록 "clean studio background" 프롬프트를 모델이 과하게 해석해 조명 장비나
추상적인 형체를 만들어내는 "오생성" 사례가 5/20건 발생했다(`docs/RESULTS.md` 참고). 배경을 비우기만
하면 되는 케이스라면 LaMa 쪽이 더 안전한 선택이었을 수 있다는 것도 이번 PoC로 확인한 한계다.

### ③ 합성
분할된 상품 픽셀과 diffusion이 생성한 배경을 페더링(feather) 마스크로 블렌딩해 경계를 자연스럽게
만든다 (`feather()` in `src/inpaint.py`).

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
