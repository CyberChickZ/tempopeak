"""
ball_finder_v11.py
基于 v10b：exit relink 使用 low-conf box 中搜索圆内最近的
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

CONF                 = 0.3
CONF_LOW             = 0.05
IMGSZ                = 960
MAX_LINK_PX          = 120
STATIC_IOU           = 0.95
STATIC_N             = 5
MAX_COAST            = 1
VEL_COAST_FACTOR     = 1.2
MIN_VEL_FOR_HIT      = 10.0
MIN_ANGLE_FOR_HIT    = 30.0
COAST_DECAY          = 0.85
NEW_TRACK_MIN_FRAMES = 3
EXIT_RELINK_PX       = 200   # 搜索圆半径，和可视化一致
EXIT_SEARCH_R        = 200
EDGE_ARM_MARGIN      = 80

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

def center(box):
    return np.array([(box[0]+box[2])/2,(box[1]+box[3])/2],dtype=float)

def shift_box(box, vel):
    x1,y1,x2,y2=box
    return [x1+vel[0],y1+vel[1],x2+vel[0],y2+vel[1]]

def is_near_edge(box, W, H, margin=EDGE_ARM_MARGIN):
    cx,cy=center(box)
    return cx<margin or cx>W-margin or cy<margin or cy>H-margin

def clamp_to_edge(box, W, H):
    cx,cy=center(box)
    return float(np.clip(cx,0,W)), float(np.clip(cy,0,H))

def boxes_overlap(a, b, min_iou=0.01):
    return iou(a,b)>min_iou

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
    candidates=[(d,float(np.linalg.norm(center(d[:4])-rc))) for d in dets]
    candidates=[(d,dist) for d,dist in candidates if dist<=max_dist]
    if not candidates: return None
    return min(candidates, key=lambda x: x[1]-squareness(x[0][:4])*5)[0]

def pick_nearest(dets, ref_center, max_dist):
    """搜索圆内最近的 box"""
    if not dets: return None
    rc=np.array(ref_center)
    candidates=[(d,float(np.linalg.norm(center(d[:4])-rc))) for d in dets]
    candidates=[(d,dist) for d,dist in candidates if dist<=max_dist]
    if not candidates: return None
    return min(candidates, key=lambda x: x[1])[0]

# ── pass 1 ────────────────────────────────────────────────────────────────────
print("Pass 1: detection...")
model=YOLO(MODEL_PATH)
cap=cv2.VideoCapture(VIDEO_PATH)
fps=cap.get(cv2.CAP_PROP_FPS)
W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

raw=[]; raw_low=[]; fi=0
while True:
    ret,frame=cap.read()
    if not ret: break

    r1=model.predict(frame,classes=[32],conf=CONF,imgsz=IMGSZ,verbose=False,device="mps")
    d1=[]
    if r1[0].boxes is not None:
        for box in r1[0].boxes:
            x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
            d1.append([float(x1),float(y1),float(x2),float(y2),float(box.conf[0])])
    raw.append(d1)

    r2=model.predict(frame,classes=[32],conf=CONF_LOW,imgsz=IMGSZ,verbose=False,device="mps")
    d2=[]
    if r2[0].boxes is not None:
        for box in r2[0].boxes:
            x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
            d2.append([float(x1),float(y1),float(x2),float(y2),float(box.conf[0])])
    raw_low.append(d2)

    fi+=1
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

# 预计算 low-conf unique boxes（不与 conf=0.3 重叠）
low_unique=[]
for fi in range(len(raw_low)):
    unique=[]
    for d_low in raw_low[fi]:
        overlaps=any(boxes_overlap(d_low[:4],d_hi[:4]) for d_hi in raw[fi])
        if not overlaps:
            unique.append(d_low)
    low_unique.append(unique)

# ── tracking ──────────────────────────────────────────────────────────────────
print("Tracking...")

trk_box           = None
trk_vel           = np.zeros(2)
trk_vel_pre_coast = np.zeros(2)
trk_coast         = 0
trk_vel_age       = 0
trk_coasting      = False
exit_center       = None
cand_box          = None
cand_vel          = np.zeros(2)
cand_frames       = 0

frame_results=[]; frame_sig=[]; frame_exit_center=[]

def reset_track():
    global trk_box,trk_vel,trk_vel_pre_coast,trk_coast,trk_vel_age,trk_coasting
    trk_box=None; trk_vel=np.zeros(2); trk_vel_pre_coast=np.zeros(2)
    trk_coast=0; trk_vel_age=0; trk_coasting=False

def reset_cand():
    global cand_box,cand_vel,cand_frames
    cand_box=None; cand_vel=np.zeros(2); cand_frames=0

for fi,dets in enumerate(filtered):
    is_pred=False; sig=0.0; chosen=None

    # ── confirmed track ───────────────────────────────────────────────────
    if trk_box is not None:
        spd=float(np.linalg.norm(trk_vel))
        if trk_coasting:
            chosen=pick_best_det(dets,trk_box,MAX_LINK_PX)
            if chosen is None:
                reset_track()
        else:
            thresh=spd*VEL_COAST_FACTOR if spd>2.0 else MAX_LINK_PX
            chosen=pick_best_det(dets,trk_box,thresh)
            if chosen is None:
                if trk_coast<MAX_COAST and spd>0.5:
                    # 先在 120px 内搜索 low-conf box
                    chosen=pick_nearest(low_unique[fi]+filtered[fi], center(trk_box), MAX_LINK_PX)
                    if chosen is None:
                        trk_vel_pre_coast=trk_vel.copy()
                        trk_coasting=True
                        trk_vel=trk_vel*COAST_DECAY
                        trk_box=shift_box(trk_box,trk_vel)
                        trk_coast+=1; is_pred=True
                    # chosen 非空则直接走下面的 update
                else:
                    reset_track()

    # ── exit relink（用 low-conf 最近 box）+ candidate ────────────────────
    if trk_box is None and chosen is None:
        # 先尝试 exit relink，用 low_unique 里搜索圆内最近的
        if exit_center is not None:
            nearest=pick_nearest(low_unique[fi] + filtered[fi], exit_center, EXIT_RELINK_PX)
            if nearest is not None:
                bc=center(nearest[:4])
                dist=float(np.linalg.norm(bc-np.array(exit_center)))
                trk_box=nearest[:4]; trk_vel=np.zeros(2)
                trk_vel_age=2; trk_coast=0; trk_coasting=False
                exit_center=None; chosen=nearest
                reset_cand()
                print(f"  f{fi:04d} EXIT RELINK ✓ dist={dist:.0f} conf={nearest[4]:.2f}")

        # 正常 candidate accumulation（conf=0.3）
        if chosen is None and dets:
            best=max(dets,key=lambda d: d[4])
            bc=center(best[:4])
            if cand_box is not None:
                dist=float(np.linalg.norm(bc-center(cand_box)))
                if dist<=MAX_LINK_PX:
                    cand_vel=bc-center(cand_box); cand_box=best[:4]; cand_frames+=1
                else:
                    cand_box=best[:4]; cand_vel=np.zeros(2); cand_frames=1
            else:
                cand_box=best[:4]; cand_vel=np.zeros(2); cand_frames=1

            if cand_frames>=NEW_TRACK_MIN_FRAMES:
                trk_box=cand_box; trk_vel=cand_vel; trk_vel_age=cand_frames
                trk_coast=0; trk_coasting=False; chosen=best
                exit_center=None; reset_cand()
                print(f"  f{fi:04d} NEW TRACK confirmed")
        elif chosen is None:
            reset_cand()

    # ── update + edge arm ─────────────────────────────────────────────────
    if chosen is not None and not is_pred:
        new_c=center(chosen[:4])
        old_c=center(trk_box) if trk_box is not None else new_c
        new_vel=new_c-old_c
        if trk_vel_age>=3:
            ref_vel=trk_vel_pre_coast if trk_coasting else trk_vel
            sig=hit_signal(ref_vel,new_vel)
        trk_vel=new_vel; trk_box=chosen[:4]
        trk_coast=0; trk_coasting=False; trk_vel_age+=1

        if is_near_edge(trk_box,W,H):
            ecx,ecy=clamp_to_edge(trk_box,W,H)
            exit_center=(ecx,ecy)
            print(f"  f{fi:04d} EDGE ARM ({ecx:.0f},{ecy:.0f})")
        else:
            exit_center=None

    frame_results.append((list(trk_box) if trk_box is not None else None,is_pred))
    frame_sig.append(sig)
    frame_exit_center.append(tuple(exit_center) if exit_center is not None else None)

    if trk_box is not None:
        sig_str=f" sig={sig:.0f}" if sig>0 else ""
        print(f"  f{fi:04d} {'P' if is_pred else 'R'} "
              f"vel={np.linalg.norm(trk_vel):.1f} age={trk_vel_age}{sig_str}")

# ── hit detection ─────────────────────────────────────────────────────────────
print("Hit detection...")
smoothed=smooth_normalize(frame_sig,SMOOTH_SIGMA)
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
    ex=frame_exit_center[fi] if fi<len(frame_exit_center) else None

    if ex is not None:
        ecx,ecy=int(ex[0]),int(ex[1])
        cv2.circle(frame,(ecx,ecy),EXIT_SEARCH_R,(255,140,0),2)
        cv2.drawMarker(frame,(ecx,ecy),(255,140,0),cv2.MARKER_CROSS,20,2)
        cv2.putText(frame,"searching",(max(0,ecx-40),min(H-10,ecy+EXIT_SEARCH_R+20)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,140,0),1)
        if fi<len(low_unique):
            for d in low_unique[fi]:
                bc=center(d[:4])
                dist=float(np.linalg.norm(bc-np.array([ecx,ecy])))
                if dist<=EXIT_SEARCH_R:
                    x1,y1,x2,y2=map(int,d[:4])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,80,0),1)
                    cv2.putText(frame,f"{d[4]:.2f}",(x1,y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,80,0),1)
            # 新增：conf=0.3 在圆内的也显示（黄色）
            for d in filtered[fi]:
                bc=center(d[:4])
                dist=float(np.linalg.norm(bc-np.array([ecx,ecy])))
                if dist<=EXIT_SEARCH_R:
                    x1,y1,x2,y2=map(int,d[:4])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),1)
                    cv2.putText(frame,f"{d[4]:.2f}",(x1,y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)

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

    status="ball" if (box is not None and not is_pred) else ("pred" if is_pred else "---")
    cv2.putText(frame,status,(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,
                (0,255,0) if status=="ball" else (180,180,180),2)
    cv2.putText(frame,f"#{fi}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    writer.write(frame); fi+=1
    if fi%50==0: print(f"\r  {fi/total*100:.0f}%",end="",flush=True)

cap.release(); writer.release()
print(f"\nDone → {OUTPUT_PATH}")