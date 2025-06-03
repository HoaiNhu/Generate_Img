import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np
import base64
import io

# Cache model globally
controlnet = None
pipe = None

def load_models():
    global controlnet, pipe
    if controlnet is None or pipe is None:
        print("Loading models...")
        with torch.no_grad():
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16  # Sử dụng half precision
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                controlnet=controlnet,
                torch_dtype=torch.float16,  # Sử dụng half precision
                safety_checker=None  # Tắt safety checker để giảm bộ nhớ
            )
            pipe = pipe.to("cpu")
            pipe.enable_attention_slicing()  # Giảm sử dụng bộ nhớ
            print("Models loaded successfully")

def process_sketch(sketch_base64):
    try:
        # Load models if not loaded
        load_models()

        # Decode base64 và resize ảnh
        sketch_data = base64.b64decode(sketch_base64.split(",")[1])
        sketch_image = Image.open(io.BytesIO(sketch_data)).convert("L")
        
        # Resize ảnh xuống 512x512 để giảm bộ nhớ
        sketch_image = sketch_image.resize((512, 512), Image.Resampling.LANCZOS)
        sketch_np = np.array(sketch_image)

        # Tiền xử lý với Canny
        sketch_canny = cv2.Canny(sketch_np, 100, 200)
        canny_image = Image.fromarray(sketch_canny)

        # Tạo hình ảnh với ít steps hơn
        prompt = "realistic cake, detailed, colorful, high quality"
        negative_prompt = "blurry, low quality, text, watermark"
        
        with torch.no_grad():  # Đảm bảo không lưu gradients
            image = pipe(
                prompt,
                image=canny_image,
                num_inference_steps=8,  # Giảm số steps
                guidance_scale=7.0,     # Giảm guidance scale
                negative_prompt=negative_prompt,
            ).images[0]

        # Chuyển kết quả thành base64 với chất lượng thấp hơn
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)  # Sử dụng JPEG thay vì PNG
        result_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return result_base64
    except Exception as e:
        print(f"Error in process_sketch: {str(e)}")
        return str(e)