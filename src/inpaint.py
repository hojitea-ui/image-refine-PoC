"""rembg 마스크를 이용해 상품은 남기고 배경만 Diffusion inpainting으로
깨끗한 스튜디오 배경으로 교체한다.

마스크 추출 로직(rembg + GrabCut 폴백)은 src/mask.py에 있고 segment.py, app.py와 공유한다.

사용법:
    python -m src.inpaint
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from src.mask import feather, get_product_mask

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/output/refined")
MODEL_ID = "runwayml/stable-diffusion-inpainting"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SD_SIZE = 512
PROMPT = "clean seamless white studio background, professional product photography, soft even lighting"
NEGATIVE_PROMPT = "shadow, clutter, text, watermark, blurry, low quality"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SD1.5 safety_checker false-positives on some benign product photos (e.g. a moon-shaped
    # lamp) and silently returns an all-black image instead of raising, corrupting the output.
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        safety_checker=None,
        requires_safety_checker=False,
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
        mask = get_product_mask(image_bgr)

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
