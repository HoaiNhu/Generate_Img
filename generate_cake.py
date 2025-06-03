import requests
import base64
import io
from PIL import Image
import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

def resize_with_padding(image, target_size=(256, 256)):
    ratio = min(target_size[0] / image.width, target_size[1] / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    
    new_image = Image.new("L", target_size, 255)
    offset = ((target_size[0] - new_size[0]) // 2,
              (target_size[1] - new_size[1]) // 2)
    new_image.paste(resized, offset)
    return new_image

def process_sketch(sketch_base64):
    try:
        # Decode base64 và resize ảnh
        sketch_data = base64.b64decode(sketch_base64.split(",")[1])
        sketch_image = Image.open(io.BytesIO(sketch_data)).convert("L")
        
        sketch_image = resize_with_padding(sketch_image, (256, 256))
        sketch_np = np.array(sketch_image)

        # Tiền xử lý với Canny
        sketch_canny = cv2.Canny(sketch_np, 100, 200)
        canny_image = Image.fromarray(sketch_canny)

        # Chuyển canny_image thành base64
        buffered = io.BytesIO()
        canny_image.save(buffered, format="PNG")
        canny_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Gọi Hugging Face API (sử dụng ControlNet Canny)
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
        }
        payload = {
            "inputs": {
                "image": canny_base64,
                "prompt": "realistic cake, detailed, colorful, high quality",
                "negative_prompt": "blurry, low quality, text, watermark",
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "controlnet_conditioning_scale": 0.8,
            },
            "options": {
                "wait_for_model": True,  # Chờ kết quả thay vì bất đồng bộ
            },
        }
        response = requests.post(
            "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_canny",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = response.json()

        # Lấy hình ảnh từ output
        image_data = result[0]  # Hugging Face trả về danh sách ảnh base64
        image = Image.open(io.BytesIO(base64.b64decode(image_data.split(",")[1])))

        # Chuyển kết quả thành base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=50)
        result_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return result_base64
    except Exception as e:
        return str(e)