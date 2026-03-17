from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import subprocess
import os
import sys
import json
import queue
import shutil
import threading
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from io_sam3 import SAM3DataStore
from io_clips import ClipStore, CLIPS_DIR, STATE_PATH

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SETTINGS_PATH = os.path.join(_PROJECT_ROOT, "data", "settings.json")

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
clip_store.scan_existing()
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
        cmd = ['osascript', '-e', 'POSIX path of (choose folder with prompt "Select Video Work Directory")']
        path = subprocess.check_output(cmd).decode('utf-8').strip()
        return {"ok": True, "path": path}
    except subprocess.CalledProcessError:
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
    hit_thresh: float = 150.0
    pad_before: int = 15
    pad_after: int = 15
    export_pad: int = 32
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    show_boxes: bool = True

class ClipAnnotateRequest(BaseModel):
    clip_id: str
    hit_frame: int

class ClipActionRequest(BaseModel):
    clip_id: str

class LastFolderRequest(BaseModel):
    folder: str


def _run_extraction(folder: str, hit_thresh: float, pad_before: int, pad_after: int,
                    export_pad: int, conf: float, iou: float, imgsz: int, show_boxes: bool):
    global _clip_processing
    _clip_processing = True

    def callback(**kwargs):
        if kwargs.get("type") == "clip":
            clip_store.add_clip(
                kwargs["clip_id"], kwargs["num_frames"],
                kwargs.get("fps", 30), kwargs.get("peak_frame", 0)
            )
        _sse_queue.put(kwargs)

    try:
        from yolo_clip_extractor import process_folder
        process_folder(folder, callback, hit_thresh=hit_thresh, pad_before=pad_before,
                       pad_after=pad_after, export_pad=export_pad, conf=conf,
                       iou=iou, imgsz=imgsz, show_boxes=show_boxes)
    except Exception as e:
        _sse_queue.put({"type": "error", "message": str(e)})
        _sse_queue.put({"type": "done"})

    _clip_processing = False


@app.post("/api/clips/start")
def clips_start(req: ClipStartRequest):
    global _clip_processing
    if _clip_processing:
        raise HTTPException(status_code=409, detail="Already processing")
    if not os.path.isdir(req.folder):
        raise HTTPException(status_code=400, detail=f"Folder not found: {req.folder}")

    # Clear existing clips from disk
    if os.path.isdir(CLIPS_DIR):
        shutil.rmtree(CLIPS_DIR)
    os.makedirs(CLIPS_DIR, exist_ok=True)

    # Reset state.json
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({}, f)

    # Reset in-memory store and SSE queue
    clip_store.reset()
    while not _sse_queue.empty():
        try:
            _sse_queue.get_nowait()
        except queue.Empty:
            break

    # Save last folder
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump({"last_folder": req.folder}, f)

    t = threading.Thread(
        target=_run_extraction,
        args=(req.folder, req.hit_thresh, req.pad_before, req.pad_after,
              req.export_pad, req.conf, req.iou, req.imgsz, req.show_boxes),
        daemon=True
    )
    t.start()
    return {"ok": True}


@app.get("/api/clips/stream")
def clips_stream():
    def gen():
        while True:
            try:
                item = _sse_queue.get(timeout=30)
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield 'data: {"type": "ping"}\n\n'
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/clips/list")
def clips_list():
    return {"clips": clip_store.get_list(), "processing": _clip_processing}


@app.get("/api/clips/frame/{clip_id}/{frame_idx}")
def clips_frame(clip_id: str, frame_idx: int):
    frame_path = os.path.join(CLIPS_DIR, clip_id, f"frame_{frame_idx:03d}.jpg")
    if not os.path.exists(frame_path):
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(frame_path, media_type="image/jpeg")


@app.get("/api/clips/meta/{clip_id}")
def clips_meta(clip_id: str):
    meta_path = os.path.join(CLIPS_DIR, clip_id, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Meta not found")
    with open(meta_path) as f:
        return json.load(f)


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


@app.post("/api/clips/clear")
def clips_clear():
    global _clip_processing
    if _clip_processing:
        raise HTTPException(status_code=409, detail="Cannot clear while processing")
    if os.path.isdir(CLIPS_DIR):
        shutil.rmtree(CLIPS_DIR)
    os.makedirs(CLIPS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({}, f)
    clip_store.reset()
    return {"ok": True}


@app.get("/api/clips/last_folder")
def get_last_folder():
    try:
        with open(SETTINGS_PATH) as f:
            s = json.load(f)
        return {"folder": s.get("last_folder", "")}
    except Exception:
        return {"folder": ""}


@app.post("/api/clips/last_folder")
def set_last_folder(req: LastFolderRequest):
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump({"last_folder": req.folder}, f)
    return {"ok": True}


# ─── Hit Annotator ───────────────────────────────────────────────────

class HAScanRequest(BaseModel):
    workdir: str

class HASaveRequest(BaseModel):
    workdir: str
    number: int
    json_data: dict
    crop_start: int = 0
    crop_end: int = 0
    fps: int = 30
    source_video: str = ""

@app.post("/api/hitannotator/scan")
def ha_scan(req: HAScanRequest):
    """Scan workdir for existing numbered .json/.mp4 files and return next available number."""
    import glob
    if not os.path.isdir(req.workdir):
        os.makedirs(req.workdir, exist_ok=True)
    max_num = 0
    for ext in ("*.json", "*.mp4"):
        for p in glob.glob(os.path.join(req.workdir, ext)):
            base = os.path.splitext(os.path.basename(p))[0]
            try:
                num = int(base)
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
    return {"next_number": max_num + 1}

@app.post("/api/hitannotator/save")
def ha_save(req: HASaveRequest):
    """Save HIT annotation JSON to workdir."""
    try:
        os.makedirs(req.workdir, exist_ok=True)
        json_name = str(req.number).zfill(5) + ".json"
        json_path = os.path.join(req.workdir, json_name)
        with open(json_path, "w") as f:
            json.dump(req.json_data, f, indent=2)
        return {"ok": True, "path": json_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hitannotator/save_video")
async def ha_save_video(workdir: str = File(...), number: int = File(...), video: UploadFile = File(...)):
    """Save uploaded video file as <number>.mp4 in workdir."""
    try:
        os.makedirs(workdir, exist_ok=True)
        video_name = str(number).zfill(5) + ".mp4"
        video_path = os.path.join(workdir, video_name)
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        return {"ok": True, "path": video_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hitannotator/save_clips")
async def ha_save_clips(
    workdir: str = Form(...),
    fps: float = Form(...),
    segments_json: str = Form(...),  # JSON: [{start, end, hits}, ...]
    video: UploadFile = File(...)
):
    """Split uploaded video into clips by segment, save each as numbered .mp4 + .json."""
    import tempfile, glob
    try:
        os.makedirs(workdir, exist_ok=True)
        segments = json.loads(segments_json)
        if not segments:
            raise HTTPException(status_code=400, detail="No segments provided")

        # Scan for next available number
        max_num = 0
        for ext in ("*.json", "*.mp4"):
            for p in glob.glob(os.path.join(workdir, ext)):
                base = os.path.splitext(os.path.basename(p))[0]
                try:
                    num = int(base)
                    if num > max_num:
                        max_num = num
                except ValueError:
                    pass
        next_num = max_num + 1

        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            shutil.copyfileobj(video.file, tmp)
            tmp_path = tmp.name

        saved = []
        try:
            for i, seg in enumerate(segments):
                num = next_num + i
                video_name = str(num).zfill(5) + ".mp4"
                json_name = str(num).zfill(5) + ".json"
                video_path = os.path.join(workdir, video_name)
                json_path = os.path.join(workdir, json_name)

                start_sec = seg["start"] / fps
                duration_sec = (seg["end"] - seg["start"]) / fps

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start_sec),
                    "-i", tmp_path,
                    "-t", str(duration_sec),
                    "-c", "copy",
                    video_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)

                json_data = {"video": video_name, "HIT": seg.get("hits", [])}
                with open(json_path, "w") as f:
                    json.dump(json_data, f, indent=2)

                saved.append({"number": num, "video": video_name})
        finally:
            os.unlink(tmp_path)

        return {"ok": True, "saved": saved}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Person Labeler ──────────────────────────────────────────────────────────

_pl_state = {"running": False, "done": 0, "total": 0, "current": ""}
_PL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolo26x.pt")
_PL_BYTETRACK  = os.path.join(os.path.dirname(__file__), "bytetrack_tennis.yaml")

class PLDetectRequest(BaseModel):
    workdir: str
    scope: str = "all"  # "all", "current", "unmodified"
    current_clip: str = None  # clip name for "current" scope

class PLSaveRequest(BaseModel):
    workdir: str
    name: str
    hits: list
    hitters: list  # [0|1|2, ...] same length as hits
    key_frames: dict  # {frame_str: {p1: detIdx|{x,y}, p2: detIdx|{x,y}}}
    frames: dict  # {frame_str: {p1: {x1,y1,x2,y2}, p2: {...}}}

class PLFillFrameRequest(BaseModel):
    workdir: str
    name: str
    frame: int
    search_box: dict  # {x1, y1, x2, y2}

def _normalize_fps(raw_fps: float) -> int:
    """Round to nearest of 15, 30, 60."""
    candidates = [15, 30, 60]
    return min(candidates, key=lambda x: abs(x - raw_fps))

def _run_pl_detection(workdir: str, scope: str = "all", current_clip: str = None):
    import glob as _glob
    import cv2 as _cv2
    global _pl_state

    all_clips = sorted(_glob.glob(os.path.join(workdir, "*.mp4")))

    # Filter clips based on scope
    if scope == "current":
        if current_clip:
            clips = [os.path.join(workdir, current_clip + ".mp4")]
        else:
            clips = []
    elif scope == "unmodified":
        # Only clips without annotations (no frames in .json)
        clips = []
        for mp4 in all_clips:
            name = os.path.splitext(os.path.basename(mp4))[0]
            json_path = os.path.join(workdir, name + ".json")
            has_annot = False
            if os.path.exists(json_path):
                try:
                    with open(json_path) as _f:
                        jd = json.load(_f)
                    frames = jd.get("frames", {})
                    has_annot = len(frames) > 0
                except Exception:
                    pass
            if not has_annot:
                clips.append(mp4)
    else:  # "all"
        clips = all_clips

    _pl_state.update({"running": True, "done": 0, "total": len(clips), "current": ""})
    try:
        from ultralytics import YOLO
        model = YOLO(_PL_MODEL_PATH)
        for mp4_path in clips:
            name = os.path.splitext(os.path.basename(mp4_path))[0]
            _pl_state["current"] = name
            det_path = os.path.join(workdir, name + "_det.json")

            # Read video metadata
            cap = _cv2.VideoCapture(mp4_path)
            raw_fps = cap.get(_cv2.CAP_PROP_FPS) or 30
            vid_w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            vid_h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
            cap.release()

            fps = _normalize_fps(raw_fps)

            frames_data = {}
            results = model.track(
                source=mp4_path, classes=[0], conf=0.1,
                tracker=_PL_BYTETRACK, stream=True,
                device="mps", half=True, vid_stride=1, verbose=False
            )
            for frame_idx, r in enumerate(results):
                if r.boxes is not None and r.boxes.id is not None:
                    dets = []
                    for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy()):
                        x1, y1, x2, y2 = map(int, box)
                        dets.append({"id": int(tid), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                    if dets:
                        frames_data[str(frame_idx)] = dets
            with open(det_path, "w") as f:
                json.dump({"fps": fps, "width": vid_w, "height": vid_h, "frames": frames_data}, f)
            _pl_state["done"] += 1
    except Exception as e:
        print(f"PL detection error: {e}")
    finally:
        _pl_state["running"] = False
        _pl_state["current"] = ""

@app.post("/api/pl/detect_start")
def pl_detect_start(req: PLDetectRequest):
    if _pl_state["running"]:
        raise HTTPException(status_code=409, detail="Already running")
    if not os.path.isdir(req.workdir):
        raise HTTPException(status_code=400, detail="Workdir not found")
    threading.Thread(target=_run_pl_detection, args=(req.workdir, req.scope, req.current_clip), daemon=True).start()
    return {"ok": True}

@app.get("/api/pl/detect_status")
def pl_detect_status():
    return _pl_state

@app.get("/api/pl/clips")
def pl_clips(workdir: str):
    import glob as _glob
    if not os.path.isdir(workdir):
        return {"clips": []}
    clips = []
    for mp4 in sorted(_glob.glob(os.path.join(workdir, "*.mp4"))):
        name = os.path.splitext(os.path.basename(mp4))[0]
        json_path = os.path.join(workdir, name + ".json")
        det_path  = os.path.join(workdir, name + "_det.json")
        json_data = {}
        if os.path.exists(json_path):
            with open(json_path) as f:
                json_data = json.load(f)
        clips.append({
            "name": name,
            "has_detection": os.path.exists(det_path),
            "hits": json_data.get("hits", json_data.get("HIT", [])),
            "key_frames": json_data.get("key_frames", {}),
            "hitters": json_data.get("hitters", []),
            "frames": json_data.get("frames", {})  # resolved boxes
        })
    return {"clips": clips}

@app.get("/api/pl/video")
def pl_video(workdir: str, name: str):
    path = os.path.join(workdir, name + ".mp4")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4")

# Frame extraction cache
_pl_frame_cache = {}
_pl_frame_cache_order = []
_PL_FRAME_CACHE_MAX = 100

def _get_cached_frame(workdir: str, name: str, frame: int):
    key = (workdir, name, frame)
    if key in _pl_frame_cache:
        return _pl_frame_cache[key]
    return None

def _cache_frame(workdir: str, name: str, frame: int, data: bytes):
    global _pl_frame_cache, _pl_frame_cache_order
    key = (workdir, name, frame)
    if key in _pl_frame_cache:
        return
    # Evict oldest if at capacity
    while len(_pl_frame_cache) >= _PL_FRAME_CACHE_MAX:
        oldest_key = _pl_frame_cache_order.pop(0)
        _pl_frame_cache.pop(oldest_key, None)
    _pl_frame_cache[key] = data
    _pl_frame_cache_order.append(key)

@app.get("/api/pl/frame")
def pl_frame(workdir: str, name: str, frame: int):
    """Extract a single frame as JPEG with LRU caching."""
    import cv2 as _cv2

    # Check cache first
    cached = _get_cached_frame(workdir, name, frame)
    if cached:
        return Response(content=cached, media_type="image/jpeg")

    path = os.path.join(workdir, name + ".mp4")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")

    cap = _cv2.VideoCapture(path)
    try:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        if not ret or img is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame} not found")
        _, buf = _cv2.imencode('.jpg', img, [_cv2.IMWRITE_JPEG_QUALITY, 85])
        data = bytes(buf)
        _cache_frame(workdir, name, frame, data)
        return Response(content=data, media_type="image/jpeg")
    finally:
        cap.release()

@app.get("/api/pl/detections")
def pl_detections(workdir: str, name: str):
    path = os.path.join(workdir, name + "_det.json")
    if not os.path.exists(path):
        return {"frames": {}}
    with open(path) as f:
        return json.load(f)

@app.post("/api/pl/save")
def pl_save(req: PLSaveRequest):
    json_path = os.path.join(req.workdir, req.name + ".json")
    # Create or update JSON file
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {"video": req.name + ".mp4"}
    data["hits"] = req.hits
    data["hitters"] = req.hitters
    data["key_frames"] = req.key_frames
    data["frames"] = req.frames
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    return {"ok": True}


class PLDeleteRequest(BaseModel):
    workdir: str
    name: str  # clip to delete, e.g. "clip_003"

@app.post("/api/pl/delete_clip")
def pl_delete_clip(req: PLDeleteRequest):
    """Delete a clip and move the last clip into the gap to keep numbering dense."""
    import glob as _glob

    workdir = req.workdir
    target = req.name

    # Enumerate all clips sorted
    all_mp4 = sorted(_glob.glob(os.path.join(workdir, "*.mp4")))
    all_names = [os.path.splitext(os.path.basename(p))[0] for p in all_mp4]

    if target not in all_names:
        raise HTTPException(status_code=404, detail=f"Clip {target} not found")

    last_name = all_names[-1]
    suffixes = [".mp4", ".json", "_det.json"]

    def remove_clip_files(name):
        for suf in suffixes:
            p = os.path.join(workdir, name + suf)
            if os.path.exists(p):
                os.remove(p)

    if target == last_name:
        # Deleting the last clip — just remove files
        remove_clip_files(target)
    else:
        # Remove target files, then rename last → target
        remove_clip_files(target)
        for suf in suffixes:
            src = os.path.join(workdir, last_name + suf)
            dst = os.path.join(workdir, target + suf)
            if os.path.exists(src):
                os.rename(src, dst)

    # Invalidate frame cache for both clips
    keys_to_remove = [k for k in _pl_frame_cache if k[1] in (target, last_name)]
    for k in keys_to_remove:
        _pl_frame_cache.pop(k, None)
        if k in _pl_frame_cache_order:
            _pl_frame_cache_order.remove(k)

    remaining = len(all_names) - 1
    return {"ok": True, "remaining": remaining, "moved": last_name if target != last_name else None}


class PLExportAllRequest(BaseModel):
    workdir: str
    export_dir: str  # user-chosen export directory

_pl_export_state = {"running": False, "done_clips": 0, "total_clips": 0,
                    "done_frames": 0, "current": "", "error": ""}

def _run_pl_export(workdir: str, export_dir: str):
    import glob as _glob
    import cv2 as _cv2
    global _pl_export_state

    video_name = os.path.basename(workdir.rstrip("/")) or "unknown"

    # Suppress cv2 h264 warnings
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

    # Find annotated clips (have .json but not _det.json)
    all_json = sorted(_glob.glob(os.path.join(workdir, "*.json")))
    clip_list = []
    for jp in all_json:
        bn = os.path.basename(jp)
        if bn.endswith("_det.json"):
            continue
        clip_name = os.path.splitext(bn)[0]
        if os.path.exists(os.path.join(workdir, clip_name + ".mp4")):
            clip_list.append((clip_name, jp))

    _pl_export_state.update({"running": True, "done_clips": 0, "total_clips": len(clip_list),
                             "done_frames": 0, "current": "", "error": ""})
    try:
        for clip_name, json_path in clip_list:
            _pl_export_state["current"] = clip_name
            video_path = os.path.join(workdir, clip_name + ".mp4")

            with open(json_path, "r") as f:
                annot_data = json.load(f)
            frames_data = annot_data.get("frames", {})
            if not frames_data:
                _pl_export_state["done_clips"] += 1
                continue

            clip_dir = os.path.join(export_dir, video_name, clip_name)
            frames_dir = os.path.join(clip_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            cap = _cv2.VideoCapture(video_path)
            if not cap.isOpened():
                _pl_export_state["done_clips"] += 1
                continue
            try:
                frame_numbers = sorted([int(k) for k in frames_data.keys()])
                for frame_num in frame_numbers:
                    cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, img = cap.read()
                    if not ret:
                        continue
                    _cv2.imwrite(os.path.join(frames_dir, f"{frame_num}.jpg"),
                                img, [_cv2.IMWRITE_JPEG_QUALITY, 95])
                    _pl_export_state["done_frames"] += 1

                annot_out = {
                    "clip": clip_name, "video": video_name,
                    "total_frames": len(frame_numbers),
                    "hits": annot_data.get("hits", []),
                    "hitters": annot_data.get("hitters", []),
                    "frames": frames_data
                }
                with open(os.path.join(clip_dir, "annot.json"), "w") as f:
                    json.dump(annot_out, f, indent=2)
            finally:
                cap.release()

            _pl_export_state["done_clips"] += 1
    except Exception as e:
        _pl_export_state["error"] = str(e)
        print(f"PL export error: {e}")
    finally:
        _pl_export_state["running"] = False
        _pl_export_state["current"] = ""

@app.post("/api/pl/export_start")
def pl_export_start(req: PLExportAllRequest):
    if _pl_export_state["running"]:
        raise HTTPException(status_code=409, detail="Export already running")
    if not os.path.isdir(req.workdir):
        raise HTTPException(status_code=400, detail="Workdir not found")
    os.makedirs(req.export_dir, exist_ok=True)
    threading.Thread(target=_run_pl_export, args=(req.workdir, req.export_dir), daemon=True).start()
    return {"ok": True}

@app.get("/api/pl/export_status")
def pl_export_status():
    return _pl_export_state


@app.post("/api/pl/fill_frame")
def pl_fill_frame(req: PLFillFrameRequest):
    """Try to detect person in a specific frame using YOLO predict (not track)."""
    import cv2 as _cv2
    from ultralytics import YOLO

    model = YOLO(_PL_MODEL_PATH)
    path = os.path.join(req.workdir, req.name + ".mp4")

    cap = _cv2.VideoCapture(path)
    cap.set(_cv2.CAP_PROP_POS_FRAMES, req.frame)
    ret, img = cap.read()
    cap.release()
    if not ret:
        return {"box": None}

    # Crop search region (padded)
    sb = req.search_box
    h, w = img.shape[:2]
    pad_w = int((sb["x2"] - sb["x1"]) * 0.5)
    pad_h = int((sb["y2"] - sb["y1"]) * 0.5)
    cx1 = max(0, int(sb["x1"]) - pad_w)
    cy1 = max(0, int(sb["y1"]) - pad_h)
    cx2 = min(w, int(sb["x2"]) + pad_w)
    cy2 = min(h, int(sb["y2"]) + pad_h)
    crop = img[cy1:cy2, cx1:cx2]

    # Run YOLO predict on crop (not track, just detect)
    results = model.predict(crop, classes=[0], conf=0.05, verbose=False, device="mps", imgsz=1280)

    if results and results[0].boxes and len(results[0].boxes) > 0:
        # Find box closest to center of search region
        best_box = None
        best_dist = float('inf')
        search_cx = (sb["x2"] + sb["x1"]) / 2 - cx1
        search_cy = (sb["y2"] + sb["y1"]) / 2 - cy1
        for box in results[0].boxes.xyxy.cpu().numpy():
            bx = (box[0] + box[2]) / 2
            by = (box[1] + box[3]) / 2
            dist = ((bx - search_cx)**2 + (by - search_cy)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_box = box
        if best_box is not None:
            # Convert crop coords back to full frame coords
            return {"box": {
                "x1": int(best_box[0] + cx1), "y1": int(best_box[1] + cy1),
                "x2": int(best_box[2] + cx1), "y2": int(best_box[3] + cy1)
            }}

    return {"box": None}


# ─── HIT Annotator (hitlabeler) ────────────────────────────────────────

class HLSaveRequest(BaseModel):
    workdir: str
    name: str
    hits: list

@app.get("/api/hl/clips")
def hl_clips(workdir: str):
    """Scan workdir for .mp4 clips, return total frames and existing HITs."""
    import cv2 as _cv2
    clips = []
    if not os.path.isdir(workdir):
        return {"clips": []}
    for f in sorted(os.listdir(workdir)):
        if not f.endswith(".mp4"):
            continue
        name = f[:-4]
        mp4_path = os.path.join(workdir, f)
        cap = _cv2.VideoCapture(mp4_path)
        total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # Check for existing json with hits
        json_path = os.path.join(workdir, name + ".json")
        hits = []
        if os.path.exists(json_path):
            try:
                with open(json_path) as jf:
                    data = json.load(jf)
                hits = data.get("hits", data.get("HIT", []))
            except:
                pass
        clips.append({"name": name, "total_frames": total, "hits": hits})
    return {"clips": clips}

@app.post("/api/hl/save")
def hl_save(req: HLSaveRequest):
    json_path = os.path.join(req.workdir, req.name + ".json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {"video": req.name + ".mp4"}
    data["hits"] = req.hits
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    return {"ok": True}


# Serve Frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
