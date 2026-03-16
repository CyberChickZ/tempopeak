"""
ball_finder_v6.py
修复：
1. 新 track 出现 (vel_age<2) 时不计 dv，避免抛球误检
2. pred 期间被击打：pred 结束后第一帧与 pred 前速度对比
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d

VIDEO_PATH  = "/Users/harryzhang/git/tempopeak/datasets/serve/00006.mp4"
OUTPUT_PATH = "/Users/harryzhang/git/tempopeak/outputs/ball_finder_v6.mp4"
MODEL_PATH  = "/Users/harryzhang/git/tempopeak/sam3_annotator/server/yolo26s.pt"

CONF             = 0.3
IMGSZ            = 960
MAX_LINK_PX      = 120
STATIC_IOU       = 0.95
STATIC_N         = 5
MAX_COAST        = 3
VEL_COAST_FACTOR = 1.2

SMOOTH_SIGMA    = 3
PEAK_MIN_DIST   = 10
PEAK_PROMINENCE = 0.25
TRAIL_LEN       = 60

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

def iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter   = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/(ua+1e-9)

def squareness(box):
    w,h = box[2]-box[0], box[3]-box[1]
    if max(w,h)<1e-9: return 0.0
    return min(w,h)/max(w,h)

def box_size(box):
    return ((box[2]-box[0])*(box[3]-box[1]))**0.5

def center(box):
    return np.array([(box[0]+box[2])/2,(box[1]+box[3])/2], dtype=float)

def shift_box(box, vel):
    x1,y1,x2,y2 = box
    return [x1+vel[0],y1+vel[1],x2+vel[0],y2+vel[1]]

def in_expanded_box(pred_box, rc, factor=1.2):
    cx=(pred_box[0]+pred_box[2])/2; cy=(pred_box[1]+pred_box[3])/2
    hw=(pred_box[2]-pred_box[0])/2*factor; hh=(pred_box[3]-pred_box[1])/2*factor
    return abs(rc[0]-cx)<=hw and abs(rc[1]-cy)<=hh

def smooth_normalize(series, sigma):
    arr=np.array(series,dtype=np.float32)
    mn,mx=arr.min(),arr.max()
    if mx-mn<1e-9: return arr
    arr=(arr-mn)/(mx-mn)
    half=int(sigma*3)
    k=np.exp(-0.5*(np.arange(-half,half+1)/sigma)**2); k/=k.sum()
    from scipy.ndimage import convolve1d
    return convolve1d(arr,k,mode="reflect")

def pick_best_det(dets, prev_box=None, prev_size=None):
    if not dets: return None
    if len(dets)==1: return dets[0]
    def score(d):
        sq=squareness(d[:4])
        dist=float(np.linalg.norm(center(d[:4])-center(prev_box))) if prev_box is not None else 0.0
        sz_diff=abs(box_size(d[:4])-prev_size)/(prev_size+1e-9) if prev_size is not None else 0.0
        return dist + sz_diff*50 - sq*10
    return min(dets, key=score)

# ── pass 1 ────────────────────────────────────────────────────────────────────
print("Pass 1: detection...")
model=YOLO(MODEL_PATH)
cap=cv2.VideoCapture(VIDEO_PATH)
fps=cap.get(cv2.CAP_PROP_FPS)
W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

raw=[]; fi=0
while True:
    ret,frame=cap.read()
    if not ret: break
    results=model.predict(frame,classes=[32],conf=CONF,imgsz=IMGSZ,verbose=False,device="mps")
    dets=[]
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
            dets.append([float(x1),float(y1),float(x2),float(y2),float(box.conf[0])])
    raw.append(dets); fi+=1
    if fi%30==0: print(f"\r  {fi/total*100:.0f}% ({fi}/{total})",end="",flush=True)
cap.release(); print(f"\n  {fi} frames")

# ── static filter ─────────────────────────────────────────────────────────────
filtered=[list(d) for d in raw]; prev_info=[]
for fi,dets in enumerate(filtered):
    keep,new_prev=[],[]
    for d in dets:
        consec=1
        for pbox,pcnt in prev_info:
            if iou(d[:4],pbox)>=STATIC_IOU: consec=pcnt+1; break
        if consec<STATIC_N: keep.append(d); new_prev.append((d[:4],consec))
    filtered[fi]=keep; prev_info=new_prev

# ── single ball tracking ──────────────────────────────────────────────────────
print("Tracking single ball...")

trk_box      = None
trk_vel      = np.zeros(2)
trk_vel_before_coast = np.zeros(2)  # vel snapshot when coast starts
trk_coast    = 0
trk_vel_age  = 0   # how many real frames since track started
trk_coasting = False

frame_results=[]; frame_dv=[]

for fi,dets in enumerate(filtered):
    chosen=None; is_pred=False; dv=0.0

    if dets:
        chosen=pick_best_det(
            dets,
            prev_box=trk_box,
            prev_size=box_size(trk_box) if trk_box is not None else None,
        )
        if trk_box is not None:
            spd=float(np.linalg.norm(trk_vel))
            thresh=spd*VEL_COAST_FACTOR if spd>2.0 else MAX_LINK_PX
            dist=float(np.linalg.norm(center(chosen[:4])-center(trk_box)))
            if dist>thresh*2: chosen=None

    if chosen is not None:
        new_c=center(chosen[:4])
        old_c=center(trk_box) if trk_box is not None else new_c
        new_vel=new_c-old_c

        if trk_coasting and trk_vel_age>=2:
            # coming back from coast: compare new vel vs vel BEFORE coast started
            dv=float(np.linalg.norm(new_vel-trk_vel_before_coast))
        elif trk_vel_age>=2:
            # normal frame-to-frame
            dv=float(np.linalg.norm(new_vel-trk_vel))
        # else: vel_age < 2 → new track, skip dv (avoid toss detection)

        trk_vel=new_vel
        trk_box=chosen[:4]
        trk_coast=0
        trk_coasting=False
        trk_vel_age+=1
        is_pred=False

    elif trk_box is not None and trk_coast<MAX_COAST and np.linalg.norm(trk_vel)>0.5:
        if not trk_coasting:
            trk_vel_before_coast=trk_vel.copy()  # snapshot before coast
            trk_coasting=True
        trk_box=shift_box(trk_box,trk_vel)
        trk_coast+=1
        is_pred=True
        chosen=trk_box+[0.0]
    else:
        trk_box=None; trk_vel=np.zeros(2)
        trk_coast=0; trk_vel_age=0; trk_coasting=False

    frame_results.append((list(trk_box) if trk_box is not None else None, is_pred))
    frame_dv.append(dv)

    if trk_box is not None:
        print(f"  f{fi:04d} {'P' if is_pred else 'R'} vel={np.linalg.norm(trk_vel):.1f} dv={dv:.1f} age={trk_vel_age}")

# ── hit detection ─────────────────────────────────────────────────────────────
print("Hit detection...")
smoothed=smooth_normalize(frame_dv,SMOOTH_SIGMA)
peaks,_=find_peaks(smoothed,distance=PEAK_MIN_DIST,
                   prominence=PEAK_PROMINENCE*(smoothed.max()-smoothed.min()))
hit_frames=set(peaks.tolist())
print(f"  {len(hit_frames)} hit candidates")

# ── pass 2: render ────────────────────────────────────────────────────────────
print("Pass 2: rendering...")
cap=cv2.VideoCapture(VIDEO_PATH)
writer=cv2.VideoWriter(OUTPUT_PATH,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
trail=[]; BAR_W=260

fi=0
while True:
    ret,frame=cap.read()
    if not ret: break
    box,is_pred=frame_results[fi] if fi<len(frame_results) else (None,False)

    if box is not None:
        x1,y1,x2,y2=map(int,box); cx,cy=(x1+x2)//2,(y1+y2)//2
        trail.append((cx,cy,is_pred))
        if len(trail)>TRAIL_LEN: trail.pop(0)
        color=(180,180,180) if is_pred else (0,255,255)
        lw=1 if is_pred else 2
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,lw)
        cv2.putText(frame,"pred" if is_pred else "ball",(x1,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,lw)
        cv2.circle(frame,(cx,cy),4,color,-1)

    for i in range(1,len(trail)):
        alpha=i/len(trail)
        c=(0,int(255*alpha),int(255*(1-alpha)))
        cv2.line(frame,trail[i-1][:2],trail[i][:2],c,1 if trail[i][2] else 2)

    dv=float(smoothed[fi]) if fi<len(smoothed) else 0.0
    x0=W-BAR_W-15
    cv2.rectangle(frame,(x0,15),(x0+BAR_W,38),(30,30,30),-1)
    cv2.rectangle(frame,(x0,15),(x0+int(dv*BAR_W),38),(0,200,255),-1)
    cv2.putText(frame,f"dv:{dv:.3f}",(x0,12),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,200,255),1)

    if fi in hit_frames:
        cv2.rectangle(frame,(0,0),(W,H),(0,0,255),10)
        cv2.putText(frame,"HIT",(W//2-60,H//2),cv2.FONT_HERSHEY_DUPLEX,3.0,(0,0,255),6)

    status="ball" if (box is not None and not is_pred) else ("pred" if is_pred else "---")
    cv2.putText(frame,status,(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,
                (0,255,0) if status=="ball" else (180,180,180),2)
    cv2.putText(frame,f"#{fi}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    writer.write(frame); fi+=1
    if fi%50==0: print(f"\r  {fi/total*100:.0f}%",end="",flush=True)

cap.release(); writer.release()
print(f"\nDone → {OUTPUT_PATH}")