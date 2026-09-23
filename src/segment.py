"""data/raw 이미지를 rembg(U2-Net)로 분할해 마스크 오버레이를 확인한다.

사용법:
    python -m src.segment
"""

from pathlib import Path

import cv2
import numpy as np

from src.mask import get_product_mask

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/output/masks")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    green = np.zeros_like(image_bgr)
    green[..., 1] = 255
    alpha = ((mask > 0).astype(np.float32) * 0.4)[..., None]
    blended = image_bgr.astype(np.float32) * (1 - alpha) + green.astype(np.float32) * alpha
    return blended.astype(np.uint8)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(p for p in RAW_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"{len(image_paths)}장 처리 시작")

    for path in image_paths:
        image_bgr = cv2.imread(str(path))
        mask = get_product_mask(image_bgr)
        area_ratio = (mask > 0).sum() / mask.size
        print(f"{path.name}: 상품 영역 비율 {area_ratio:.1%}")

        overlay = overlay_mask(image_bgr, mask)
        out_path = OUTPUT_DIR / f"{path.stem}_mask.jpg"
        cv2.imwrite(str(out_path), overlay)

    print(f"\n완료: {len(image_paths)}장, 결과는 {OUTPUT_DIR}/ 에 저장")


if __name__ == "__main__":
    main()
