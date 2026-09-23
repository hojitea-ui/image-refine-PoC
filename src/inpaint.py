"""YOLOv8-seg 마스크를 이용해 상품은 남기고 배경만 Diffusion inpainting으로
깨끗한 스튜디오 배경으로 교체한다.

segment.py가 먼저 실행되어 data/output/masks_raw/*.npy (마스크 배열)를 만들어둔
상태를 전제로 한다. 이 스크립트는 segment.py의 마스크 생성 로직을 재사용해
바로 원본 이미지에서 마스크를 뽑아 이어서 처리한다.

사용법:
    python src/inpaint.py
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from ultralytics import YOLO

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/output/refined")
WEIGHTS = Path("weights/yolov8n-seg.pt")
MODEL_ID = "runwayml/stable-diffusion-inpainting"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SD_SIZE = 512
PROMPT = "clean seamless white studio background, professional product photography, soft even lighting"
NEGATIVE_PROMPT = "shadow, clutter, text, watermark, blurry, low quality"


MIN_AREA_RATIO = 0.04
MAX_AREA_RATIO = 0.65


def grabcut_fallback(image_bgr: np.ndarray) -> np.ndarray:
    """YOLO 마스크가 못 미더울 때(상품이 COCO 80클래스에 없어 탐지 실패) 쓰는
    classical CV 폴백. 화면 중앙 60% 영역을 전경으로 가정하고 GrabCut을 돌린다."""
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    margin_x, margin_y = int(w * 0.2), int(h * 0.2)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return fg_mask


def get_product_mask(result, image_bgr: np.ndarray) -> np.ndarray:
    """YOLO 결과에서 상품으로 신뢰할 만한 마스크를 고르고, 없으면 GrabCut으로 대체한다.

    상품 후보(효도박스, 스피커 등)는 COCO 80개 클래스에 없는 경우가 많아
    YOLO가 아예 탐지를 못 하거나 배경의 잡동사니를 잘못 고르는 경우가 있다.
    화면 대비 너무 작거나(잡동사니) 너무 큰(배경 오탐) 후보는 걸러내고,
    남은 후보가 없으면 GrabCut 폴백으로 넘어간다.
    """
    h, w = result.orig_shape
    image_area = h * w

    if result.masks is not None:
        areas = result.masks.data.sum(dim=(1, 2))
        candidates = [
            i
            for i in range(len(areas))
            if MIN_AREA_RATIO * image_area <= areas[i].item() <= MAX_AREA_RATIO * image_area
        ]
        if candidates:
            top = max(candidates, key=lambda i: areas[i].item())
            mask = result.masks.data[top].cpu().numpy()
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            return (mask > 0.5).astype(np.uint8) * 255

    return grabcut_fallback(image_bgr)


def feather(mask: np.ndarray, ksize: int = 9) -> np.ndarray:
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(WEIGHTS))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    image_paths = sorted(p for p in RAW_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"{len(image_paths)}장 대상 inpainting 시작 (device: {device})")

    for path in image_paths:
        out_path = OUTPUT_DIR / f"{path.stem}_refined.jpg"
        if out_path.exists():
            print(f"{path.name} -> {out_path.name} (skip, 이미 존재)")
            continue

        image_bgr = cv2.imread(str(path))
        result = yolo.predict(source=str(path), device=0 if device == "cuda" else "cpu", verbose=False)[0]
        mask = get_product_mask(result, image_bgr)

        orig = Image.open(path).convert("RGB")
        orig_w, orig_h = orig.size

        orig_sq = orig.resize((SD_SIZE, SD_SIZE))
        inpaint_mask = 255 - mask  # 상품(255) 제외 배경만 255=inpaint 대상
        inpaint_mask_sq = cv2.resize(inpaint_mask, (SD_SIZE, SD_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_img = Image.fromarray(inpaint_mask_sq)

        generated = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=orig_sq,
            mask_image=mask_img,
            num_inference_steps=25,
        ).images[0]

        generated_full = generated.resize((orig_w, orig_h))

        product_mask_soft = feather(mask).astype(np.float32) / 255.0
        product_mask_soft = product_mask_soft[..., None]

        orig_arr = np.array(orig, dtype=np.float32)
        gen_arr = np.array(generated_full, dtype=np.float32)
        composite = orig_arr * product_mask_soft + gen_arr * (1 - product_mask_soft)
        composite = composite.clip(0, 255).astype(np.uint8)

        out_path = OUTPUT_DIR / f"{path.stem}_refined.jpg"
        Image.fromarray(composite).save(out_path, quality=95)
        print(f"{path.name} -> {out_path.name}")

    print(f"\n완료: {len(image_paths)}/{len(image_paths)}, 결과는 {OUTPUT_DIR}/ 에 저장")


if __name__ == "__main__":
    main()
