"""자가 촬영 상품 사진 배경/조명 정제 데모.

사진을 업로드하면 rembg로 상품을 분할하고, Diffusion inpainting으로
배경을 깨끗한 스튜디오 배경으로 교체한 결과를 보여준다.

사용법:
    python app.py
"""

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from src.inpaint import MODEL_ID, NEGATIVE_PROMPT, PROMPT, SD_SIZE
from src.mask import feather, get_product_mask

_device = "cuda" if torch.cuda.is_available() else "cpu"
# safety_checker false-positives on some benign product photos and silently returns an
# all-black image instead of raising, so it's disabled here too (see src/inpaint.py).
_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
    variant="fp16" if _device == "cuda" else None,
    safety_checker=None,
    requires_safety_checker=False,
).to(_device)
_pipe.set_progress_bar_config(disable=True)


def refine(image: Image.Image):
    if image is None:
        return None, None

    orig = image.convert("RGB")
    orig_w, orig_h = orig.size
    image_bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)

    mask = get_product_mask(image_bgr)

    mask_preview = Image.fromarray(mask)

    orig_sq = orig.resize((SD_SIZE, SD_SIZE))
    inpaint_mask = 255 - mask
    inpaint_mask_sq = cv2.resize(inpaint_mask, (SD_SIZE, SD_SIZE), interpolation=cv2.INTER_NEAREST)
    mask_img = Image.fromarray(inpaint_mask_sq)

    generated = _pipe(
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

    return mask_preview, Image.fromarray(composite)


demo = gr.Interface(
    fn=refine,
    inputs=gr.Image(type="pil", label="상품 촬영 사진 업로드"),
    outputs=[
        gr.Image(type="pil", label="rembg 분할 마스크"),
        gr.Image(type="pil", label="배경 정제 결과"),
    ],
    title="자가 촬영 상품 사진 배경/조명 정제",
    description=(
        "직접 촬영한 상품 사진에서 rembg로 상품 영역을 분할하고, "
        "Diffusion inpainting으로 배경을 깨끗한 스튜디오 배경으로 교체합니다."
    ),
)

if __name__ == "__main__":
    demo.launch()
