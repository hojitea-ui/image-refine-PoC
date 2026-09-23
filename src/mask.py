"""rembg(U2-Net)로 상품 전경 마스크를 뽑는다. 실패 시 GrabCut으로 대체한다.

segment.py, inpaint.py, app.py가 공통으로 쓰는 마스크 추출 로직.
"""

import cv2
import numpy as np
from rembg import new_session, remove

MIN_AREA_RATIO = 0.04
MAX_AREA_RATIO = 0.65

_session = new_session("u2net")


def grabcut_fallback(image_bgr: np.ndarray) -> np.ndarray:
    """rembg 마스크가 못 미더울 때(전경 비율이 너무 작거나 큼) 쓰는 classical CV 폴백.
    화면 중앙 60% 영역을 전경으로 가정하고 GrabCut을 돌린다."""
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    margin_x, margin_y = int(w * 0.2), int(h * 0.2)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return fg_mask


def get_product_mask(image_bgr: np.ndarray) -> np.ndarray:
    """rembg로 상품 마스크를 뽑고, 전경 비율이 비정상적으로 작거나 크면 GrabCut으로 대체한다."""
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = remove(image_rgb, session=_session, only_mask=True)
    mask = np.array(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    area_ratio = (mask > 0).sum() / (h * w)
    if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
        return grabcut_fallback(image_bgr)
    return mask


def feather(mask: np.ndarray, ksize: int = 9) -> np.ndarray:
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)
