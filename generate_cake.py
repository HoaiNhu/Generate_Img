import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np
import base64
import io

def process_sketch(sketch_base64):
    try:
        # Decode base64
        sketch_data = base64.b64decode(sketch_base64.split(",")[1])
        sketch_image = Image.open(io.BytesIO(sketch_data)).convert("L")
        sketch_np = np.array(sketch_image)

        # Tiền xử lý với Canny
        sketch_canny = cv2.Canny(sketch_np, 100, 200)
        canny_image = Image.fromarray(sketch_canny)

        # Load mô hình
        with torch.no_grad():
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                controlnet=controlnet,
                torch_dtype=torch.float32
            )
            pipe = pipe.to("cpu")

        # Tạo hình ảnh
        prompt = "realistic cake, detailed, colorful, high quality"
        negative_prompt = "blurry, low quality, text, watermark"
        image = pipe(
            prompt,
            image=canny_image,
            num_inference_steps=10,
            guidance_scale=7.5,
            negative_prompt=negative_prompt,
        ).images[0]

        # Chuyển kết quả thành base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        result_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return result_base64
    except Exception as e:
        return str(e)