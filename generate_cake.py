import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np
import base64
import io
import gc

# Cache model globally
controlnet = None
pipe = None

def load_models():
    global controlnet, pipe
    if controlnet is None or pipe is None:
        print("Loading models...")
        try:
            # Giải phóng bộ nhớ trước khi load model mới
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            with torch.no_grad():
                # Sử dụng model nhẹ hơn
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny",
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                )
                
                # Sử dụng model base thay vì 2.1
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",  # Model nhẹ hơn
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    safety_checker=None
                )
                
                # Tối ưu hóa pipeline
                pipe = pipe.to("cpu")
                pipe.enable_attention_slicing(slice_size="auto")
                pipe.enable_model_cpu_offload()  # Offload model ra CPU khi không cần
                pipe.enable_vae_slicing()  # Tối ưu VAE
                
                print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise e

def resize_with_padding(image, target_size=(192, 192)):  # Giảm xuống 192x192
    """Resize ảnh giữ nguyên tỷ lệ và thêm padding"""
    ratio = min(target_size[0] / image.width, target_size[1] / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Tạo ảnh mới với padding
    new_image = Image.new("L", target_size, 255)
    offset = ((target_size[0] - new_size[0]) // 2,
             (target_size[1] - new_size[1]) // 2)
    new_image.paste(resized, offset)
    return new_image

def process_sketch(sketch_base64):
    try:
        # Load models if not loaded
        load_models()

        # Decode base64 và resize ảnh
        sketch_data = base64.b64decode(sketch_base64.split(",")[1])
        sketch_image = Image.open(io.BytesIO(sketch_data)).convert("L")
        
        # Resize ảnh xuống 192x192 với padding
        sketch_image = resize_with_padding(sketch_image, (192, 192))
        sketch_np = np.array(sketch_image)

        # Tiền xử lý với Canny
        sketch_canny = cv2.Canny(sketch_np, 100, 200)
        canny_image = Image.fromarray(sketch_canny)

        # Tạo hình ảnh với ít steps hơn
        prompt = "realistic cake, detailed, colorful, high quality"
        negative_prompt = "blurry, low quality, text, watermark"
        
        with torch.no_grad():
            # Giải phóng bộ nhớ trước khi generate
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            image = pipe(
                prompt,
                image=canny_image,
                num_inference_steps=4,  # Giảm xuống 4 steps
                guidance_scale=6.0,     # Giảm guidance scale
                negative_prompt=negative_prompt,
            ).images[0]

            # Giải phóng bộ nhớ sau khi generate
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Chuyển kết quả thành base64 với chất lượng thấp hơn
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=75)  # Giảm chất lượng xuống 75%
        result_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return result_base64
    except Exception as e:
        print(f"Error in process_sketch: {str(e)}")
        # Giải phóng bộ nhớ trong trường hợp lỗi
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return str(e)