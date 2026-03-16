"""
ball_finder_v11.py
状态机: IDLE → TRACKING → LOST → DEAD(→IDLE)
- TRACKING/LOST 都只在预测位置 crop 内搜索，conf=0.15
- IDLE 用全图 conf=0.3
- LOST 搜索半径固定上限，不随帧数增长
- hit_signal = angle（速度只做门槛）
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d

VIDEO_PATH  = "/Users/harryzhang/git/tempopeak/datasets/serve/00011.mp4"
OUTPUT_PATH = "/Users/harryzhang/git/tempopeak/outputs/ball_finder_v11.mp4"
MODEL_PATH  = "/Users/harryzhang/git/tempopeak/sam3_annotator/server/yolo26s.pt"

# detection
CONF_IDLE        = 0.30
CONF_ACTIVE      = 0.15   # TRACKING + LOST 阶段 crop 内用
IMGSZ            = 960

# tracking
SEARCH_R_TRACKING  = 80    # px，TRACKING 阶段 crop 半径
SEARCH_R_LOST      = 200   # px，LOST 阶段固定上限
MAX_LOST_FRAMES    = 15    # 超过才 → DEAD
STATIC_IOU         = 0.95
STATIC_N           = 5
NEW_TRACK_MIN_FRAMES = 3
COAST_DECAY        = 0.90  # LOST 阶段速度衰减/帧

# hit
MIN_VEL_FOR_HIT    = 10.0
MIN_ANGLE_FOR_HIT  = 30.0

# render
SMOOTH_SIGMA    = 3
PEAK_MIN_DIST   = 10
PEAK_PROMINENCE = 0.25
TRAIL_LEN       = 60

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def iou(a, b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1])
    ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/(ua+1e-9)

def squareness(box):
    w,h=box[2]-box[0],box[3]-box[1]
    if max(w,h)<1e-9: return 0.0
    return min(w,h)/max(w,h)

def center(box):
    return np.array([(box[0]+box[2])/2,(box[1]+box[3])/2],dtype=float)

def shift_box(box, vel):
    x1,y1,x2,y2=box; return [x1+vel[0],y1+vel[1],x2+vel[0],y2+vel[1]]

def hit_signal(v1, v2):
    n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
    if n1<0.5 or n2<MIN_VEL_FOR_HIT: return 0.0
    cos_a=np.clip(np.dot(v1,v2)/(n1*n2),-1.0,1.0)
    angle=float(np.degrees(np.arccos(cos_a)))
    return angle if angle>=MIN_ANGLE_FOR_HIT else 0.0

def smooth_normalize(series, sigma):
    arr=np.array(series,dtype=np.float32)
    mn,mx=arr.min(),arr.max()
    if mx-mn<1e-9: return arr
    arr=(arr-mn)/(mx-mn)
    half=int(sigma*3)
    k=np.exp(-0.5*(np.arange(-half,half+1)/sigma)**2); k/=k.sum()
    return convolve1d(arr,k,mode="reflect")

def pick_best_det(dets, ref_box, max_dist):
    if not dets: return None
    rc=center(ref_box)
    cands=[(d,float(np.linalg.norm(center(d[:4])-rc))) for d in dets]
    cands=[(d,dist) for d,dist in cands if dist<=max_dist]
    if not cands: return None
    return min(cands, key=lambda x: x[1]-squareness(x[0][:4])*5)[0]

def detect_in_crop(frame, pred_box, search_r, model, W, H):
    """pred_box 中心附近 crop，低 conf 检测，返回全图坐标的 dets"""
    cx,cy=center(pred_box)
    x1c=max(0,int(cx-search_r)); y1c=max(0,int(cy-search_r))
    x2c=min(W,int(cx+search_r)); y2c=min(H,int(cy+search_r))
    if x2c<=x1c or y2c<=y1c: return []
    crop=frame[y1c:y2c, x1c:x2c]
    res=model.predict(crop,classes=[32],conf=CONF_ACTIVE,
                      imgsz=IMGSZ,verbose=False,device="mps")
    dets=[]
    if res[0].boxes is not None:
        for box in res[0].boxes:
            bx1,by1,bx2,by2=box.xyxy[0].cpu().numpy()
            dets.append([float(bx1+x1c),float(by1+y1c),
                         float(bx2+x1c),float(by2+y1c),
                         float(box.conf[0])])
    return dets

# ── pass 1: read frames ───────────────────────────────────────────────────────
print("Pass 1: loading frames...")
cap=cv2.VideoCapture(VIDEO_PATH)
fps=cap.get(cv2.CAP_PROP_FPS)
W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frames=[]
while True:
    ret,frame=cap.read()
    if not ret: break
    frames.append(frame)
    if len(frames)%30==0:
        print(f"\r  {len(frames)/total*100:.0f}%",end="",flush=True)
cap.release(); print(f"\n  {len(frames)} frames")

# ── global detection pass (IDLE conf) ────────────────────────────────────────
print("Pass 2: global detection...")
model=YOLO(MODEL_PATH)
raw=[]
for fi,frame in enumerate(frames):
    res=model.predict(frame,classes=[32],conf=CONF_IDLE,
                      imgsz=IMGSZ,verbose=False,device="mps")
    dets=[]
    if res[0].boxes is not None:
        for box in res[0].boxes:
            x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
            dets.append([float(x1),float(y1),float(x2),float(y2),float(box.conf[0])])
    raw.append(dets)
    if fi%30==0: print(f"\r  {fi/total*100:.0f}%",end="",flush=True)
print(f"\n  done")

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

# ── state machine tracking ────────────────────────────────────────────────────
print("Tracking...")

# states
IDLE=0; TRACKING=1; LOST=2
state       = IDLE

trk_box     = None
trk_vel     = np.zeros(2)
trk_vel_age = 0
lost_frames = 0

# candidate buffer (IDLE only)
cand_box    = None
cand_vel    = np.zeros(2)
cand_frames = 0

frame_results=[]; frame_sig=[]

for fi,frame in enumerate(frames):
    dets_global = filtered[fi]
    is_pred=False; sig=0.0; chosen=None

    # ── TRACKING ──────────────────────────────────────────────────────────
    if state==TRACKING:
        pred_box=shift_box(trk_box, trk_vel)
        dets=detect_in_crop(frame, pred_box, SEARCH_R_TRACKING, model, W, H)
        chosen=pick_best_det(dets, pred_box, SEARCH_R_TRACKING)

        if chosen is None:
            # 没找到 → LOST，保留当前预测位置
            state=LOST; lost_frames=1
            trk_box=pred_box
            trk_vel=trk_vel*COAST_DECAY
            is_pred=True
            print(f"  f{fi:04d} → LOST")
        else:
            new_vel=center(chosen[:4])-center(trk_box)
            if trk_vel_age>=3:
                sig=hit_signal(trk_vel, new_vel)
            trk_vel=new_vel; trk_box=chosen[:4]; trk_vel_age+=1

    # ── LOST ──────────────────────────────────────────────────────────────
    elif state==LOST:
        pred_box=shift_box(trk_box, trk_vel)
        dets=detect_in_crop(frame, pred_box, SEARCH_R_LOST, model, W, H)
        chosen=pick_best_det(dets, pred_box, SEARCH_R_LOST)

        if chosen is not None:
            new_vel=center(chosen[:4])-center(trk_box)
            if trk_vel_age>=3:
                sig=hit_signal(trk_vel, new_vel)
            trk_vel=new_vel; trk_box=chosen[:4]; trk_vel_age+=1
            state=TRACKING; lost_frames=0
            print(f"  f{fi:04d} → TRACKING (reacquired after {lost_frames} lost)")
        else:
            lost_frames+=1
            trk_box=pred_box
            trk_vel=trk_vel*COAST_DECAY
            is_pred=True
            if lost_frames>=MAX_LOST_FRAMES:
                state=IDLE
                trk_box=None; trk_vel=np.zeros(2); trk_vel_age=0
                cand_box=None; cand_vel=np.zeros(2); cand_frames=0
                print(f"  f{fi:04d} → DEAD → IDLE")

    # ── IDLE ──────────────────────────────────────────────────────────────
    elif state==IDLE:
        if dets_global:
            best=max(dets_global,key=lambda d: d[4])
            bc=center(best[:4])
            if cand_box is not None:
                dist=float(np.linalg.norm(bc-center(cand_box)))
                if dist<=120:
                    cand_vel=bc-center(cand_box)
                    cand_box=best[:4]; cand_frames+=1
                else:
                    cand_box=best[:4]; cand_vel=np.zeros(2); cand_frames=1
            else:
                cand_box=best[:4]; cand_vel=np.zeros(2); cand_frames=1

            if cand_frames>=NEW_TRACK_MIN_FRAMES:
                trk_box=cand_box; trk_vel=cand_vel; trk_vel_age=cand_frames
                state=TRACKING
                cand_box=None; cand_vel=np.zeros(2); cand_frames=0
                chosen=best
                print(f"  f{fi:04d} → TRACKING (new track)")
        else:
            cand_box=None; cand_vel=np.zeros(2); cand_frames=0

    frame_results.append((list(trk_box) if trk_box is not None else None, is_pred))
    frame_sig.append(sig)

    if trk_box is not None:
        sig_str=f" sig={sig:.0f}" if sig>0 else ""
        print(f"  f{fi:04d} [{['IDLE','TRK','LOST'][state]}] "
              f"{'P' if is_pred else 'R'} "
              f"vel={np.linalg.norm(trk_vel):.1f} age={trk_vel_age}{sig_str}")

# ── hit detection ─────────────────────────────────────────────────────────────
print("Hit detection...")
smoothed=smooth_normalize(frame_sig,SMOOTH_SIGMA)
peaks,_=find_peaks(smoothed,distance=PEAK_MIN_DIST,
                   prominence=PEAK_PROMINENCE*(smoothed.max()-smoothed.min()))
hit_frames=set(peaks.tolist())
print(f"  {len(hit_frames)} hit candidates: {sorted(hit_frames)}")

# ── pass 3: render ────────────────────────────────────────────────────────────
print("Pass 3: rendering...")
writer=cv2.VideoWriter(OUTPUT_PATH,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
trail=[]; BAR_W=260
STATE_COLORS={IDLE:(128,128,128),TRACKING:(0,255,255),LOST:(0,165,255)}

# rebuild state per frame for rendering
# (re-run state from frame_results is enough)
render_state=[]
s=IDLE
for fi,(box,ip) in enumerate(frame_results):
    if box is not None and not ip: s=TRACKING
    elif box is not None and ip: s=LOST
    else: s=IDLE
    render_state.append(s)

for fi,frame in enumerate(frames):
    box,is_pred=frame_results[fi]
    st=render_state[fi]
    col=STATE_COLORS[st]

    if box is not None:
        x1,y1,x2,y2=map(int,box); cx,cy=(x1+x2)//2,(y1+y2)//2
        trail.append((cx,cy,is_pred))
        if len(trail)>TRAIL_LEN: trail.pop(0)
        lw=1 if is_pred else 2
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,lw)
        cv2.putText(frame,"pred" if is_pred else "ball",(x1,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,col,lw)
        cv2.circle(frame,(cx,cy),4,col,-1)

    for i in range(1,len(trail)):
        alpha=i/len(trail)
        c=(0,int(255*alpha),int(255*(1-alpha)))
        cv2.line(frame,trail[i-1][:2],trail[i][:2],c,
                 1 if trail[i][2] else 2)

    sv=float(smoothed[fi]) if fi<len(smoothed) else 0.0
    x0=W-BAR_W-15
    cv2.rectangle(frame,(x0,15),(x0+BAR_W,38),(30,30,30),-1)
    cv2.rectangle(frame,(x0,15),(x0+int(sv*BAR_W),38),(0,200,255),-1)
    cv2.putText(frame,f"sig:{sv:.3f}",(x0,12),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,200,255),1)

    if fi in hit_frames:
        cv2.rectangle(frame,(0,0),(W,H),(0,0,255),10)
        cv2.putText(frame,"HIT",(W//2-60,H//2),
                    cv2.FONT_HERSHEY_DUPLEX,3.0,(0,0,255),6)

    state_str=["IDLE","TRK","LOST"][st]
    cv2.putText(frame,state_str,(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
    cv2.putText(frame,f"#{fi}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    writer.write(frame)
    if fi%50==0: print(f"\r  {fi/total*100:.0f}%",end="",flush=True)

writer.release()
print(f"\nDone → {OUTPUT_PATH}")