"""
ball_finder_v7.py
pred 逻辑：最多 1 帧，pred 后 120px 内任意方向搜索
hit 逻辑：vel_age>=3 且 vel>=10px/frame 才算 dv
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d

VIDEO_PATH  = "/Users/harryzhang/git/tempopeak/datasets/serve/00006.mp4"
OUTPUT_PATH = "/Users/harryzhang/git/tempopeak/outputs/ball_finder_v7.mp4"
MODEL_PATH  = "/Users/harryzhang/git/tempopeak/sam3_annotator/server/yolo26s.pt"

CONF             = 0.3
IMGSZ            = 960
MAX_LINK_PX      = 120
STATIC_IOU       = 0.95
STATIC_N         = 5
MAX_COAST        = 1
VEL_COAST_FACTOR = 1.2
MIN_VEL_FOR_HIT  = 10.0   # px/frame，低于此速度的 dv 不算 hit（过滤抛球）

SMOOTH_SIGMA    = 3
PEAK_MIN_DIST   = 10
PEAK_PROMINENCE = 0.25
TRAIL_LEN       = 60

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

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

def box_size(box):
    return ((box[2]-box[0])*(box[3]-box[1]))**0.5

def center(box):
    return np.array([(box[0]+box[2])/2,(box[1]+box[3])/2],dtype=float)

def shift_box(box, vel):
    x1,y1,x2,y2=box
    return [x1+vel[0],y1+vel[1],x2+vel[0],y2+vel[1]]

def smooth_normalize(series, sigma):
    arr=np.array(series,dtype=np.float32)
    mn,mx=arr.min(),arr.max()
    if mx-mn<1e-9: return arr
    arr=(arr-mn)/(mx-mn)
    half=int(sigma*3)
    k=np.exp(-0.5*(np.arange(-half,half+1)/sigma)**2); k/=k.sum()
    return convolve1d(arr,k,mode="reflect")

def pick_best_det(dets, ref_box, max_dist):
    """max_dist 内找最近+最圆的检测"""
    if not dets: return None
    rc=center(ref_box)
    candidates=[(d,float(np.linalg.norm(center(d[:4])-rc))) for d in dets]
    candidates=[(d,dist) for d,dist in candidates if dist<=max_dist]
    if not candidates: return None
    return min(candidates, key=lambda x: x[1]-squareness(x[0][:4])*5)[0]

# ── pass 1: detect ────────────────────────────────────────────────────────────
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
    if fi%30==0: print(f"\r  {fi/total*100:.0f}%",end="",flush=True)
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

# ── single ball state machine ─────────────────────────────────────────────────
print("Tracking...")

trk_box           = None
trk_vel           = np.zeros(2)
trk_vel_pre_coast = np.zeros(2)
trk_coast         = 0
trk_vel_age       = 0
trk_coasting      = False

frame_results=[]; frame_dv=[]

for fi,dets in enumerate(filtered):
    is_pred=False; dv=0.0; chosen=None

    if trk_box is not None:
        spd=float(np.linalg.norm(trk_vel))

        if trk_coasting:
            # pred 后：120px 内任意方向
            chosen=pick_best_det(dets, trk_box, MAX_LINK_PX)
        else:
            # 正常：速度方向阈值内
            thresh=spd*VEL_COAST_FACTOR if spd>2.0 else MAX_LINK_PX
            chosen=pick_best_det(dets, trk_box, thresh)

        if chosen is None:
            if not trk_coasting and trk_coast<MAX_COAST and spd>0.5:
                # 第一次丢失 → pred 一帧
                trk_vel_pre_coast=trk_vel.copy()
                trk_coasting=True
                trk_box=shift_box(trk_box,trk_vel)
                trk_coast+=1
                is_pred=True
            else:
                # pred 后还是没找到 → 丢失
                trk_box=None; trk_vel=np.zeros(2)
                trk_coast=0; trk_vel_age=0; trk_coasting=False
    else:
        # 无 track：接受最高 conf 新检测
        if dets:
            chosen=max(dets, key=lambda d: d[4])

    if chosen is not None:
        new_c  =center(chosen[:4])
        old_c  =center(trk_box) if trk_box is not None else new_c
        new_vel=new_c-old_c

        if trk_vel_age>=3:
            if trk_coasting:
                # coast 恢复：与 coast 前速度对比
                dv=float(np.linalg.norm(new_vel-trk_vel_pre_coast))
            else:
                dv=float(np.linalg.norm(new_vel-trk_vel))

            # 低速阶段不算 hit（抛球、慢速运动）
            if float(np.linalg.norm(new_vel))<MIN_VEL_FOR_HIT:
                dv=0.0

        trk_vel     =new_vel
        trk_box     =chosen[:4]
        trk_coast   =0
        trk_coasting=False
        trk_vel_age +=1
        is_pred     =False

    frame_results.append((list(trk_box) if trk_box is not None else None, is_pred))
    frame_dv.append(dv)

    if trk_box is not None:
        print(f"  f{fi:04d} {'P' if is_pred else 'R'} "
              f"vel={np.linalg.norm(trk_vel):.1f} dv={dv:.1f} age={trk_vel_age}")

# ── hit detection ─────────────────────────────────────────────────────────────
print("Hit detection...")
smoothed=smooth_normalize(frame_dv,SMOOTH_SIGMA)
peaks,_=find_peaks(smoothed,distance=PEAK_MIN_DIST,
                   prominence=PEAK_PROMINENCE*(smoothed.max()-smoothed.min()))
hit_frames=set(peaks.tolist())
print(f"  {len(hit_frames)} hit candidates: {sorted(hit_frames)}")

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
        cv2.putText(frame,"HIT",(W//2-60,H//2),
                    cv2.FONT_HERSHEY_DUPLEX,3.0,(0,0,255),6)

    status="ball" if (box is not None and not is_pred) else ("pred" if is_pred else "---")
    cv2.putText(frame,status,(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,
                (0,255,0) if status=="ball" else (180,180,180),2)
    cv2.putText(frame,f"#{fi}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    writer.write(frame); fi+=1
    if fi%50==0: print(f"\r  {fi/total*100:.0f}%",end="",flush=True)

cap.release(); writer.release()
print(f"\nDone → {OUTPUT_PATH}")