from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
import subprocess
import os
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from io_sam3 import SAM3DataStore

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_store = SAM3DataStore()

class EditRequest(BaseModel):
    frame_idx: int
    object_id: str
    prompt: str

class DeleteRequest(BaseModel):
    frame_idx: int
    object_id: str

class DeleteTrackRequest(BaseModel):
    object_id: str

class ScanRequest(BaseModel):
    workdir: str

class LoadRequest(BaseModel):
    video_name: str

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/browse_dir")
def browse_dir():
    try:
        # Calls native macOS folder picker
        cmd = ['osascript', '-e', 'POSIX path of (choose folder with prompt "Select Video Work Directory")']
        path = subprocess.check_output(cmd).decode('utf-8').strip()
        return {"ok": True, "path": path}
    except subprocess.CalledProcessError:
        # Canceled by user
        return {"ok": False, "path": ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan_dir")
def scan_dir(req: ScanRequest):
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        videos = data_store.scan_directory(req.workdir, base_dir)
        return {"ok": True, "videos": videos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load_by_name")
def load_by_name(req: LoadRequest):
    try:
        data_store.load_from_name(req.video_name)
        return {"ok": True, "message": f"Loaded {req.video_name} successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video")
def get_video():
    if not data_store.video_path or not os.path.exists(data_store.video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    # For a robust implementation, streaming response with range queries is better, 
    # but FileResponse works fine for local development and simple MP4s.
    return FileResponse(data_store.video_path, media_type="video/mp4")

@app.post("/api/upload")
async def upload_files(
    json_file: UploadFile = File(...),
    npz_file: UploadFile = File(...)
):
    try:
        json_bytes = await json_file.read()
        npz_bytes = await npz_file.read()
        data_store.load_from_bytes(json_bytes, npz_bytes)
        return {"ok": True, "message": "Files loaded successfully into memory"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/frame/{frame_idx}")
def get_frame(frame_idx: int):
    # Returns exactly what's in track.json for that frame
    # v2 schema: {"0": {"label": "ball", "tracker_score": ..., "box_xyxy": ...}, ...}
    # v1 schema: {"0": {"prompt": "ball", "score": ..., "box": ...}, ...}
    instances = data_store.get_frame_instances(frame_idx)
    return {"frame_idx": frame_idx, "instances": instances}

@app.get("/api/mask/{mask_idx}.png")
def get_mask(mask_idx: int):
    png_bytes = data_store.get_mask_png_bytes(mask_idx)
    if not png_bytes:
        raise HTTPException(status_code=404, detail="Mask not found")
    return Response(content=png_bytes, media_type="image/png")

@app.post("/api/edit")
def edit_instance(req: EditRequest):
    success = data_store.edit_label(req.frame_idx, req.object_id, req.prompt)
    if not success:
        raise HTTPException(status_code=404, detail="Instance not found")
    return {"ok": True}

@app.post("/api/delete")
def delete_instance(req: DeleteRequest):
    success = data_store.delete_instance(req.frame_idx, req.object_id)
    if not success:
        raise HTTPException(status_code=404, detail="Instance not found")
    return {"ok": True}

@app.post("/api/delete_track")
def delete_track(req: DeleteTrackRequest):
    success = data_store.delete_track(req.object_id)
    if not success:
        raise HTTPException(status_code=404, detail="Track not found")
    return {"ok": True}

@app.post("/api/save_overwrite")
def save_overwrite():
    success = data_store.save_overwrite()
    if not success:
        raise HTTPException(status_code=400, detail="Cannot save. No active file opened.")
    return {"ok": True}

@app.get("/api/download_json")
def download_json():
    if data_store.masks_array is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    json_bytes = data_store.generate_download_json()
    return Response(
        content=json_bytes, 
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=edited.json"}
    )

@app.get("/api/download_npz")
def download_npz():
    if data_store.masks_array is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    npz_bytes = data_store.generate_download_npz()
    return Response(
        content=npz_bytes, 
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=edited.npz"}
    )

# Serve Frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
