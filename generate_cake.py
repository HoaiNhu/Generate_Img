import requests
import base64
import io
from PIL import Image
import cv2
import numpy as np
import os
from dotenv import load_dotenv
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
if not HF_API_TOKEN:
    logger.error("HF_API_TOKEN không được cấu hình trong biến môi trường")

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
        logger.info("Bắt đầu xử lý sketch")
        
        # Decode base64 và resize ảnh
        try:
            sketch_data = base64.b64decode(sketch_base64.split(",")[1])
            sketch_image = Image.open(io.BytesIO(sketch_data)).convert("L")
            logger.info("Đã decode và convert ảnh thành công")
        except Exception as e:
            logger.error(f"Lỗi khi decode/convert ảnh: {str(e)}")
            raise
        
        try:
            sketch_image = resize_with_padding(sketch_image, (256, 256))
            sketch_np = np.array(sketch_image)
            logger.info("Đã resize ảnh thành công")
        except Exception as e:
            logger.error(f"Lỗi khi resize ảnh: {str(e)}")
            raise

        # Tiền xử lý với Canny
        try:
            sketch_canny = cv2.Canny(sketch_np, 100, 200)
            canny_image = Image.fromarray(sketch_canny)
            logger.info("Đã xử lý Canny thành công")
        except Exception as e:
            logger.error(f"Lỗi khi xử lý Canny: {str(e)}")
            raise

        # Chuyển canny_image thành base64
        try:
            buffered = io.BytesIO()
            canny_image.save(buffered, format="PNG")
            canny_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            logger.info("Đã chuyển ảnh Canny thành base64")
        except Exception as e:
            logger.error(f"Lỗi khi chuyển ảnh Canny thành base64: {str(e)}")
            raise

        # Gọi Hugging Face API
        try:
            if not HF_API_TOKEN:
                raise ValueError("HF_API_TOKEN không tồn tại")
                
            logger.info("Đang gọi Hugging Face API...")
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
                    "wait_for_model": True,
                },
            }
            
            response = requests.post(
                "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_canny",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            logger.info("Đã nhận kết quả từ Hugging Face API")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Lỗi khi gọi Hugging Face API: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response text: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Lỗi không xác định khi gọi API: {str(e)}")
            raise

        # Xử lý kết quả
        try:
            image_data = result[0]
            image = Image.open(io.BytesIO(base64.b64decode(image_data.split(",")[1])))
            
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=50)
            result_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            logger.info("Đã xử lý và trả về kết quả thành công")
            return result_base64
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý kết quả từ API: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Lỗi tổng thể trong process_sketch: {str(e)}")
        return str(e)