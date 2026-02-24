import io
import os
import json
import numpy as np
import PIL.Image as Image

class SAM3DataStore:
    def __init__(self):
        self.tracks = {} 
        self.masks_array = None
        self.frame_indices = None
        self.object_ids = None
        self.video_path = None
        self.current_json_path = None
        self.current_npz_path = None
        self.available_videos = {}

    def scan_directory(self, workdir: str, base_dir: str):
        self.available_videos = {}
        if not os.path.exists(workdir) or not os.path.isdir(workdir):
            raise FileNotFoundError(f"Directory not found: {workdir}")
            
        out_dir = os.path.join(base_dir, "outputs", "sam3_mask_extractor")
        
        for file in sorted(os.listdir(workdir)):
            if file.lower().endswith(".mp4"):
                video_name = os.path.splitext(file)[0]
                video_path = os.path.join(workdir, file)
                
                # Rule 1: same directory
                json_path = os.path.join(workdir, f"{video_name}.json")
                npz_path = os.path.join(workdir, f"{video_name}.npz")
                
                # Rule 2: tempopeak out dir
                if not (os.path.exists(json_path) and os.path.exists(npz_path)):
                    json_path = os.path.join(out_dir, f"{video_name}.json")
                    npz_path = os.path.join(out_dir, f"{video_name}.npz")
                    
                if os.path.exists(json_path) and os.path.exists(npz_path):
                    self.available_videos[video_name] = {
                        "video": video_path,
                        "json": json_path,
                        "npz": npz_path
                    }
        return list(self.available_videos.keys())

    def load_from_name(self, video_name: str):
        if video_name not in self.available_videos:
            raise FileNotFoundError(f"Video '{video_name}' not found. Please scan directory first.")
            
        info = self.available_videos[video_name]
        self.video_path = info["video"]
        self.current_json_path = info["json"]
        self.current_npz_path = info["npz"]
        
        with open(self.current_json_path, 'r', encoding='utf-8') as f:
            raw_tracks = json.load(f)
            self.tracks = {int(k): v for k, v in raw_tracks.items()}
            
        with open(self.current_npz_path, 'rb') as f:
            data = np.load(f)
            self.masks_array = data['masks']
            self.frame_indices = data['frame_indices']
            self.object_ids = data['object_ids']

    def load_from_bytes(self, json_bytes: bytes, npz_bytes: bytes):
        raw_tracks = json.loads(json_bytes.decode('utf-8'))
        self.tracks = {int(k): v for k, v in raw_tracks.items()}
        
        with io.BytesIO(npz_bytes) as f:
            data = np.load(f)
            self.masks_array = data['masks']
            self.frame_indices = data['frame_indices']
            self.object_ids = data['object_ids']

    def get_frame_instances(self, frame_idx: int):
        return self.tracks.get(frame_idx, {})
        
    def get_mask_png_bytes(self, mask_idx: int) -> bytes:
        if self.masks_array is None or mask_idx < 0 or mask_idx >= len(self.masks_array):
            return b""
        mask_2d = self.masks_array[mask_idx]
        
        # Create an RGBA image: white color where mask is true, fully transparent elsewhere
        img_array = np.zeros((*mask_2d.shape, 4), dtype=np.uint8)
        img_array[mask_2d] = [255, 255, 255, 255]
        
        img = Image.fromarray(img_array, mode='RGBA')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def edit_label(self, frame_idx: int, obj_id: str, new_prompt: str):
        if frame_idx in self.tracks and obj_id in self.tracks[frame_idx]:
            self.tracks[frame_idx][obj_id]['prompt'] = new_prompt
            return True
        return False
        
    def delete_instance(self, frame_idx: int, obj_id: str):
        if frame_idx in self.tracks and obj_id in self.tracks[frame_idx]:
            del self.tracks[frame_idx][obj_id]
            return True
        return False
        
    def delete_track(self, obj_id: str):
        deleted_any = False
        for frame_idx, frame_data in self.tracks.items():
            if obj_id in frame_data:
                del self.tracks[frame_idx][obj_id]
                deleted_any = True
        return deleted_any

    def generate_download_json(self) -> bytes:
        stringified = {str(k): v for k, v in self.tracks.items()}
        return json.dumps(stringified, indent=2).encode('utf-8')

    def generate_download_npz(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            masks=self.masks_array,
            frame_indices=self.frame_indices,
            object_ids=self.object_ids
        )
        return buf.getvalue()

    def save_overwrite(self) -> bool:
        if not self.current_json_path or not self.current_npz_path:
            return False
            
        # Overwrite JSON
        json_bytes = self.generate_download_json()
        with open(self.current_json_path, 'wb') as f:
            f.write(json_bytes)
            
        # Overwrite NPZ
        npz_bytes = self.generate_download_npz()
        with open(self.current_npz_path, 'wb') as f:
            f.write(npz_bytes)
            
        return True
