"""
Robi Graphics Upscaling - FastAPI Backend
Real-ESRGAN দিয়ে আসল AI আপস্কেলিং

Install:
  pip install fastapi uvicorn python-multipart pillow basicsr realesrgan

Run:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import os
import uuid
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

# ── Real-ESRGAN imports ──────────────────────────────────────────────────────
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    print("⚠️  Real-ESRGAN not installed. pip install basicsr realesrgan")

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Robi Graphics Upscaling API",
    description="Real-ESRGAN দিয়ে AI Image Upscaling",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path(tempfile.gettempdir()) / "robi_upscaled"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Model Cache ──────────────────────────────────────────────────────────────
_upsampler_cache: dict = {}

def get_upsampler(scale: int = 4, model_name: str = "RealESRGAN_x4plus"):
    """Real-ESRGAN model লোড করে cache করে রাখে"""
    cache_key = f"{model_name}_{scale}"
    if cache_key in _upsampler_cache:
        return _upsampler_cache[cache_key]

    if not REALESRGAN_AVAILABLE:
        raise RuntimeError("Real-ESRGAN installed নেই!")

    # মডেল কনফিগ
    model_configs = {
        "RealESRGAN_x4plus": {
            "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                             num_block=23, num_grow_ch=32, scale=4),
            "scale": 4,
            "model_path": "weights/RealESRGAN_x4plus.pth",
            "tile": 400,
        },
        "RealESRGAN_x2plus": {
            "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                             num_block=23, num_grow_ch=32, scale=2),
            "scale": 2,
            "model_path": "weights/RealESRGAN_x2plus.pth",
            "tile": 400,
        },
        "RealESRGAN_x4plus_anime_6B": {
            "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                             num_block=6, num_grow_ch=32, scale=4),
            "scale": 4,
            "model_path": "weights/RealESRGAN_x4plus_anime_6B.pth",
            "tile": 400,
        },
    }

    cfg = model_configs.get(model_name, model_configs["RealESRGAN_x4plus"])

    # Model weights ডাউনলোড করা না থাকলে auto-download
    weight_path = cfg["model_path"]
    if not os.path.exists(weight_path):
        os.makedirs("weights", exist_ok=True)
        download_weights(model_name, weight_path)

    upsampler = RealESRGANer(
        scale=cfg["scale"],
        model_path=weight_path,
        model=cfg["model"],
        tile=cfg["tile"],
        tile_pad=10,
        pre_pad=0,
        half=False,  # GPU থাকলে True করুন
    )

    _upsampler_cache[cache_key] = upsampler
    return upsampler


def download_weights(model_name: str, save_path: str):
    """Model weights না থাকলে ডাউনলোড করে"""
    import urllib.request

    urls = {
        "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    }

    url = urls.get(model_name)
    if not url:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"📥 Downloading {model_name} weights...")
    urllib.request.urlretrieve(url, save_path)
    print(f"✅ Downloaded to {save_path}")


# ── PIL Fallback Upscaler (Real-ESRGAN না থাকলে) ────────────────────────────
def pil_upscale(img: Image.Image, scale: int) -> Image.Image:
    """Pillow দিয়ে Lanczos upscaling (fallback)"""
    w, h = img.size
    new_w, new_h = w * scale, h * scale
    return img.resize((new_w, new_h), Image.LANCZOS)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app": "Robi Graphics Upscaling",
        "realesrgan": REALESRGAN_AVAILABLE,
        "status": "ready"
    }


@app.get("/health")
def health():
    return {"status": "ok", "realesrgan_available": REALESRGAN_AVAILABLE}


@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    scale: int = Form(default=4),
    model: str = Form(default="RealESRGAN_x4plus"),
    output_format: str = Form(default="png"),
):
    """
    ছবি আপলোড করুন → AI দিয়ে আপস্কেল হবে → ডাউনলোড লিংক পাবেন

    - **file**: ইমেজ ফাইল (JPG/PNG/WEBP)
    - **scale**: 2, 4, বা 8 (default: 4)
    - **model**: RealESRGAN_x4plus | RealESRGAN_x2plus | RealESRGAN_x4plus_anime_6B
    - **output_format**: png | jpg | webp
    """

    # ── Validation ──
    if scale not in [2, 4, 8]:
        raise HTTPException(400, "scale অবশ্যই 2, 4, বা 8 হতে হবে")

    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"শুধু JPG/PNG/WEBP সাপোর্ট করে। আপনি দিয়েছেন: {file.content_type}")

    # ── Read Image ──
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"ইমেজ পড়া যাচ্ছে না: {str(e)}")

    orig_w, orig_h = pil_img.size

    # ── Upscale ──
    try:
        if REALESRGAN_AVAILABLE:
            # Real-ESRGAN দিয়ে আসল AI আপস্কেলিং
            # scale=8 হলে 2x দুইবার চালাই
            if scale == 8:
                upsampler = get_upsampler(4, "RealESRGAN_x4plus")
                img_array = np.array(pil_img)
                output_array, _ = upsampler.enhance(img_array, outscale=4)
                upsampler2 = get_upsampler(2, "RealESRGAN_x2plus")
                output_array, _ = upsampler2.enhance(output_array, outscale=2)
            else:
                model_map = {2: "RealESRGAN_x2plus", 4: "RealESRGAN_x4plus"}
                model_name = model if model != "auto" else model_map.get(scale, "RealESRGAN_x4plus")
                upsampler = get_upsampler(scale, model_name)
                img_array = np.array(pil_img)
                output_array, _ = upsampler.enhance(img_array, outscale=scale)

            result_img = Image.fromarray(output_array)
        else:
            # Fallback: Pillow Lanczos
            result_img = pil_upscale(pil_img, scale)

    except Exception as e:
        raise HTTPException(500, f"Upscaling এ সমস্যা হয়েছে: {str(e)}")

    # ── Save Output ──
    out_name = f"Robi_Graphics_{uuid.uuid4().hex[:8]}_{scale}x.{output_format}"
    out_path = OUTPUT_DIR / out_name

    fmt_map = {"png": "PNG", "jpg": "JPEG", "webp": "WEBP"}
    save_fmt = fmt_map.get(output_format, "PNG")

    save_kwargs = {}
    if save_fmt == "JPEG":
        save_kwargs["quality"] = 95
    elif save_fmt == "WEBP":
        save_kwargs["quality"] = 90
        save_kwargs["lossless"] = False

    result_img.save(out_path, format=save_fmt, **save_kwargs)

    new_w, new_h = result_img.size

    return JSONResponse({
        "success": True,
        "download_url": f"/download/{out_name}",
        "filename": out_name,
        "original_resolution": f"{orig_w}×{orig_h}",
        "upscaled_resolution": f"{new_w}×{new_h}",
        "scale": scale,
        "model_used": "Real-ESRGAN" if REALESRGAN_AVAILABLE else "Pillow Lanczos (fallback)",
        "file_size_kb": round(os.path.getsize(out_path) / 1024, 1),
    })


@app.get("/download/{filename}")
def download_file(filename: str):
    """আপস্কেল হওয়া ছবি ডাউনলোড করুন — auto download হবে"""
    # Security: path traversal ঠেকানো
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "Invalid filename")

    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "ফাইল পাওয়া যায়নি। আবার আপস্কেল করুন।")

    # Content-Disposition: attachment → browser auto-download করবে
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-cache",
        }
    )


@app.get("/models")
def list_models():
    """Available upscaling models"""
    return {
        "models": [
            {
                "id": "RealESRGAN_x4plus",
                "name": "Real-ESRGAN x4+ (General)",
                "scales": [4],
                "best_for": "ফটো, landscape, portrait"
            },
            {
                "id": "RealESRGAN_x2plus",
                "name": "Real-ESRGAN x2+ (Fast)",
                "scales": [2],
                "best_for": "দ্রুত আপস্কেলিং"
            },
            {
                "id": "RealESRGAN_x4plus_anime_6B",
                "name": "Real-ESRGAN Anime x4",
                "scales": [4],
                "best_for": "Anime, cartoon, illustration"
            },
        ]
    }


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
