import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
from PIL import Image
import cv2
import numpy as np
import base64
import io
import gc
import os

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

            # Giảm số lượng workers của torch
            torch.set_num_threads(1)
            os.environ["OMP_NUM_THREADS"] = "1"

            with torch.no_grad():
                # Sử dụng model nhẹ nhất có thể
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    use_safetensors=False  # Không sử dụng safetensors để giảm bộ nhớ
                )
                
                # Sử dụng model base nhẹ nhất
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    "CompVis/stable-diffusion-v1-4",  # Model nhẹ hơn v1-5
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    use_safetensors=False,
                    safety_checker=None
                )
                
                # Tối ưu hóa pipeline
                pipe = pipe.to("cpu")
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)  # Scheduler nhanh hơn
                pipe.enable_attention_slicing(slice_size=1)  # Slice size nhỏ nhất
                pipe.enable_model_cpu_offload()
                pipe.enable_vae_slicing()
                pipe.enable_vae_tiling()  # Thêm tiling để giảm bộ nhớ
                
                print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise e

def resize_with_padding(image, target_size=(64, 64)):  # Giảm xuống 64x64
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
        
        # Resize ảnh xuống 64x64 với padding
        sketch_image = resize_with_padding(sketch_image, (64, 64))
        sketch_np = np.array(sketch_image)

        # Tiền xử lý với Canny
        sketch_canny = cv2.Canny(sketch_np, 100, 200)
        canny_image = Image.fromarray(sketch_canny)

        # Tạo hình ảnh với ít steps nhất có thể
        prompt = "cake, colorful"  # Rút gọn prompt
        negative_prompt = "blurry, text"  # Rút gọn negative prompt
        
        with torch.no_grad():
            # Giải phóng bộ nhớ trước khi generate
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Generate với ít steps nhất
            image = pipe(
                prompt,
                image=canny_image,
                num_inference_steps=1,  # Giảm xuống 1 step
                guidance_scale=3.0,     # Giảm guidance scale
                negative_prompt=negative_prompt,
            ).images[0]

            # Giải phóng bộ nhớ sau khi generate
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Chuyển kết quả thành base64 với chất lượng thấp nhất
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=50)  # Giảm chất lượng xuống 50%
        result_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return result_base64
    except Exception as e:
        print(f"Error in process_sketch: {str(e)}")
        # Giải phóng bộ nhớ trong trường hợp lỗi
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return str(e)