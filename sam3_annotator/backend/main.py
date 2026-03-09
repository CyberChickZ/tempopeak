from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import subprocess
import os
import sys
import json
import queue
import threading
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from io_sam3 import SAM3DataStore
from io_clips import ClipStore

_scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts"))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_store = SAM3DataStore()
clip_store = ClipStore()
_sse_queue: queue.Queue = queue.Queue()
_clip_processing: bool = False

class EditRequest(BaseModel):
    frame_idx: int
    object_id: str
    prompt: str

class EditTrackRequest(BaseModel):
    object_id: str
    prompt: str

class DeleteRequest(BaseModel):
    frame_idx: int
    object_id: str

class DeleteTrackRequest(BaseModel):
    object_id: str

class EditHitScoreRequest(BaseModel):
    frame_idx: int
    sigma: float = 2.5

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

@app.post("/api/edit_track")
def edit_track(req: EditTrackRequest):
    success = data_store.edit_track_label(req.object_id, req.prompt)
    if not success:
        raise HTTPException(status_code=404, detail="Track not found or could not be edited")
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

@app.post("/api/edit_hit_score_gaussian")
def edit_hit_score_gaussian(req: EditHitScoreRequest):
    success = data_store.apply_hit_score_gaussian(req.frame_idx, req.sigma)
    if not success:
        raise HTTPException(status_code=400, detail="Could not apply hit scores")
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

# ─── Clip Review ─────────────────────────────────────────────────────

class ClipStartRequest(BaseModel):
    folder: str
    model: str = "yolo26x.pt"
    conf: float = 0.25

class ClipAnnotateRequest(BaseModel):
    clip_id: str
    hit_frame: int

class ClipActionRequest(BaseModel):
    clip_id: str

def _run_extraction(folder, model_name, conf):
    global _clip_processing
    _clip_processing = True
    tmp_dir = "/tmp/clips"
    os.makedirs(tmp_dir, exist_ok=True)
    def callback(**kwargs):
        if kwargs.get("type") == "clip":
            clip_store.add_clip(kwargs["clip_id"], kwargs["path"], kwargs["num_frames"])
        _sse_queue.put(kwargs)
    try:
        from yolo_clip_extractor import process_folder
        process_folder(folder, tmp_dir, callback, model_name=model_name, conf=conf)
    except Exception as e:
        _sse_queue.put({"type": "error", "message": str(e)})
    _clip_processing = False
    _sse_queue.put({"type": "done"})

@app.post("/api/clips/start")
def clips_start(req: ClipStartRequest):
    global _clip_processing
    if _clip_processing:
        raise HTTPException(status_code=409, detail="Already processing")
    if not os.path.isdir(req.folder):
        raise HTTPException(status_code=400, detail=f"Folder not found: {req.folder}")
    clip_store.reset()
    while not _sse_queue.empty():
        try: _sse_queue.get_nowait()
        except queue.Empty: break
    t = threading.Thread(target=_run_extraction, args=(req.folder, req.model, req.conf), daemon=True)
    t.start()
    return {"ok": True}

@app.get("/api/clips/stream")
def clips_stream():
    def gen():
        while True:
            try:
                item = _sse_queue.get(timeout=30)
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") in ("done", "error"): break
            except queue.Empty:
                yield 'data: {"type": "ping"}\n\n'
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/api/clips/video/{clip_id}")
def clips_video(clip_id: str):
    path = clip_store.get_clip_path(clip_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(path, media_type="video/mp4")

@app.get("/api/clips/list")
def clips_list():
    return {"ok": True, "clips": clip_store.get_all()}

@app.post("/api/clips/annotate")
def clips_annotate(req: ClipAnnotateRequest):
    if not clip_store.annotate(req.clip_id, req.hit_frame):
        raise HTTPException(status_code=400, detail="Invalid clip_id or hit_frame")
    return {"ok": True}

@app.post("/api/clips/reject")
def clips_reject(req: ClipActionRequest):
    clip_store.reject(req.clip_id)
    return {"ok": True}

@app.post("/api/clips/undo")
def clips_undo(req: ClipActionRequest):
    clip_store.undo(req.clip_id)
    return {"ok": True}

@app.post("/api/clips/export")
def clips_export():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/final"))
    result = clip_store.export(out_dir)
    return {"ok": True, **result}

# Serve Frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
