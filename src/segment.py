"""data/raw 이미지를 YOLOv8-seg로 분할해 마스크 오버레이를 확인한다.

사용법:
    python src/segment.py
"""

from pathlib import Path

import torch
from ultralytics import YOLO

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/output/masks")
WEIGHTS = Path("weights/yolov8n-seg.pt")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS))
    image_paths = sorted(p for p in RAW_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"{len(image_paths)}장 처리 시작 (device: {device})")

    no_mask = []
    for path in image_paths:
        result = model.predict(source=str(path), device=device, verbose=False)[0]

        if result.masks is None:
            no_mask.append(path.name)
            print(f"[분할 실패] {path.name}: 마스크 없음")
            continue

        areas = result.masks.data.sum(dim=(1, 2))
        top = areas.argmax().item()
        cls_name = result.names[int(result.boxes.cls[top])]
        conf = result.boxes.conf[top].item()
        print(f"{path.name}: {len(result.masks)}개 객체, 최상위={cls_name} ({conf:.2f})")

        overlay = result.plot()
        out_path = OUTPUT_DIR / f"{path.stem}_mask.jpg"
        import cv2

        cv2.imwrite(str(out_path), overlay)

    print(f"\n완료: {len(image_paths) - len(no_mask)}/{len(image_paths)} 성공, 결과는 {OUTPUT_DIR}/ 에 저장")
    if no_mask:
        print(f"마스크 없음: {no_mask}")


if __name__ == "__main__":
    main()
