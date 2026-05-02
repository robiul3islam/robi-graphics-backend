import io
import os
import uuid
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="Robi Graphics Upscaling API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path(tempfile.gettempdir()) / "robi_upscaled"
OUTPUT_DIR.mkdir(exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok", "realesrgan_available": False}

@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    scale: int = Form(default=4),
    model: str = Form(default="auto"),
    output_format: str = Form(default="png"),
):
    from PIL import Image

    if scale not in [2, 4, 8]:
        raise HTTPException(400, "scale অবশ্যই 2, 4, বা 8 হতে হবে")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"ইমেজ পড়া যাচ্ছে না: {str(e)}")

    orig_w, orig_h = img.size
    result = img.resize((orig_w * scale, orig_h * scale), Image.LANCZOS)

    out_name = f"Robi_Graphics_upscaling_{uuid.uuid4().hex[:8]}_{scale}x.{output_format}"
    out_path = OUTPUT_DIR / out_name

    fmt_map = {"png": "PNG", "jpg": "JPEG", "webp": "WEBP"}
    result.save(out_path, format=fmt_map.get(output_format, "PNG"))

    new_w, new_h = result.size

    return JSONResponse({
        "success": True,
        "download_url": f"/download/{out_name}",
        "filename": out_name,
        "original_resolution": f"{orig_w}×{orig_h}",
        "upscaled_resolution": f"{new_w}×{new_h}",
        "scale": scale,
        "model_used": "Pillow Lanczos",
        "file_size_kb": round(os.path.getsize(out_path) / 1024, 1),
    })

@app.get("/download/{filename}")
def download_file(filename: str):
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "Invalid filename")
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "ফাইল পাওয়া যায়নি")
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
