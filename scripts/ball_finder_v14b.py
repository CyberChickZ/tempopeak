"""
ball_finder_v14b.py
sig = real_acc × turn_sig × smoothness
  heading: 垂直向上=0°，顺时针，0-360
  turn_deg: 最小合理夹角 |Δheading| ∈ [0,180]，无smooth，纯 t-2→t-1 vs t-1→t
  smoothness: exp(-σ_heading/30)，只在这个term上加smooth
  real_acc: 透视归一化加速度（sqrt(ref_area/area)*px_vel）
面积比<0.4 或 |rv2|<10 直接 sig=0
hit: sig >= MIN_SIG_THRESH 的连续簇取第一帧
渲染: 全帧显示所有 low_unique + filtered box（常驻）
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

VIDEO_PATH  = "/Users/harryzhang/git/tempopeak/datasets/serve/00011.mp4"
OUTPUT_PATH = "/Users/harryzhang/git/tempopeak/outputs/ball_finder_v14b.mp4"
MODEL_PATH  = "/Users/harryzhang/git/tempopeak/sam3_annotator/server/yolo26s.pt"

CONF                 = 0.3
CONF_LOW             = 0.05
IMGSZ                = 960
MAX_LINK_PX          = 120
STATIC_IOU           = 0.95
STATIC_N             = 5
MAX_COAST            = 1
VEL_COAST_FACTOR     = 1.2
COAST_DECAY          = 0.85
NEW_TRACK_MIN_FRAMES = 3
EXIT_RELINK_PX       = 200
EXIT_SEARCH_R        = 200
EDGE_ARM_MARGIN      = 80
MIN_VEL_FOR_HIT      = 10.0   # rv2 归一化后最低速度
MIN_AREA_RATIO       = 0.4    # box面积/ref_area 低于此拒绝
MIN_SIG_THRESH       = 3.0    # hit 门槛
SMOOTH_HALFLIFE_DEG  = 30.0   # smoothness exp 半衰常数（°）
TRAIL_LEN            = 60

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

# ── 工具 ──────────────────────────────────────────────────────────────────────
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

def box_area(box):
    return max(0.0,(box[2]-box[0])*(box[3]-box[1]))

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

def heading(v):
    """垂直向上=0°，顺时针，[0,360)"""
    deg = float(np.degrees(np.arctan2(v[0], -v[1]))) % 360.0
    return deg

def min_turn_deg(v1, v2):
    """最小合理夹角 ∈ [0,180]，无smooth"""
    n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
    if n1<1e-9 or n2<1e-9: return 0.0
    h1,h2=heading(v1),heading(v2)
    delta=((h2-h1+180)%360)-180   # [-180,180]
    return abs(delta)

def pick_best_det(dets, ref_box, max_dist):
    if not dets: return None
    rc=center(ref_box)
    candidates=[(d,float(np.linalg.norm(center(d[:4])-rc))) for d in dets]
    candidates=[(d,dist) for d,dist in candidates if dist<=max_dist]
    if not candidates: return None
    return min(candidates, key=lambda x: x[1]-squareness(x[0][:4])*5)[0]

def pick_nearest(dets, ref_center, max_dist):
    if not dets: return None
    rc=np.array(ref_center)
    candidates=[(d,float(np.linalg.norm(center(d[:4])-rc))) for d in dets]
    candidates=[(d,dist) for d,dist in candidates if dist<=max_dist]
    if not candidates: return None
    return min(candidates, key=lambda x: x[1])[0]

def compute_sig(v1, v2, box, ref_area, vel_history):
    """
    v1 = trk_vel_prev (t-2→t-1)
    v2 = new_vel      (t-1→t)
    box = 当前帧 box
    vel_history = 近N帧 px_vel list（不含v2）
    返回 (sig, info_dict)
    """
    area  = box_area(box)
    ratio = area / max(ref_area, 1.0)

    # 面积硬 gate
    if ratio < MIN_AREA_RATIO:
        return 0.0, {"skip": f"area_ratio={ratio:.2f}<{MIN_AREA_RATIO}",
                     "area": f"{area:.0f}", "ratio": f"{ratio:.2f}"}

    # 透视归一化
    scale = float(np.sqrt(ref_area / max(area, 1.0)))
    rv1   = v1 * scale
    rv2   = v2 * scale
    rn1   = float(np.linalg.norm(rv1))
    rn2   = float(np.linalg.norm(rv2))

    # 速度硬 gate
    if rn1 < MIN_VEL_FOR_HIT or rn2 < MIN_VEL_FOR_HIT:
        return 0.0, {"skip": f"|rv1|={rn1:.1f} or |rv2|={rn2:.1f}<{MIN_VEL_FOR_HIT}",
                     "scale": f"{scale:.2f}", "area": f"{area:.0f}",
                     "ratio": f"{ratio:.2f}", "rn1": f"{rn1:.1f}", "rn2": f"{rn2:.1f}"}

    # A: real_acc（归一化加速度幅度）
    real_acc = float(np.linalg.norm(rv2 - rv1))

    # B: turn_sig — 纯 t-2→t-1 vs t-1→t，无任何 smooth
    turn_d   = min_turn_deg(v1, v2)           # [0,180]
    h1       = heading(v1)
    h2       = heading(v2)
    turn_sig = (1.0 - np.cos(np.radians(turn_d))) / 2.0  # sin²(θ/2)

    # C: smoothness — heading 历史标准差，exp 衰减
    #    vel_history 存原始 px vel（不缩放），只用方向
    headings = [heading(v) for v in vel_history if np.linalg.norm(v)>1e-3]
    if len(headings) >= 2:
        # circular std：先转 unit vector 再算
        sins = np.sin(np.radians(headings))
        coss = np.cos(np.radians(headings))
        mean_sin, mean_cos = sins.mean(), coss.mean()
        R = np.sqrt(mean_sin**2 + mean_cos**2)   # 0=随机, 1=完全一致
        sigma_deg = float(np.degrees(np.sqrt(-2*np.log(max(R,1e-9)))))
    else:
        sigma_deg = 0.0
    smoothness = float(np.exp(-sigma_deg / SMOOTH_HALFLIFE_DEG))

    sig = real_acc * turn_sig * smoothness

    info = {
        "skip"      : None,
        "area"      : f"{area:.0f}",
        "ratio"     : f"{ratio:.2f}",
        "scale"     : f"{scale:.2f}",
        "rn1"       : f"{rn1:.1f}",
        "rn2"       : f"{rn2:.1f}",
        "h1"        : f"{h1:.1f}°",
        "h2"        : f"{h2:.1f}°",
        "turn"      : f"{turn_d:.1f}°",
        "turn_sig"  : f"{turn_sig:.3f}",
        "sigma_hdg" : f"{sigma_deg:.1f}°",
        "smooth"    : f"{smoothness:.3f}",
        "real_acc"  : f"{real_acc:.1f}",
        "sig"       : f"{sig:.2f}",
    }
    return sig, info

# ── pass 1: detection ─────────────────────────────────────────────────────────
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

low_unique=[]
for fi in range(len(raw_low)):
    unique=[]
    for d_low in raw_low[fi]:
        overlaps=any(boxes_overlap(d_low[:4],d_hi[:4]) for d_hi in raw[fi])
        if not overlaps: unique.append(d_low)
    low_unique.append(unique)

# ref_area
all_areas=[box_area(d[:4]) for frame in raw for d in frame if box_area(d[:4])>0]
ref_area=float(np.median(all_areas)) if all_areas else 400.0
print(f"  ref_area={ref_area:.0f}px² (n={len(all_areas)})")

# ── tracking ──────────────────────────────────────────────────────────────────
print("Tracking...")

trk_box      = None
trk_vel      = np.zeros(2)
trk_vel_prev = np.zeros(2)
trk_coast    = 0
trk_vel_age  = 0
trk_coasting = False
exit_center  = None
exit_vel     = np.zeros(2)
relink_frame = False
cand_box     = None
cand_vel     = np.zeros(2)
cand_frames  = 0
vel_history  = []   # 近6帧 px_vel，给 smoothness 用

frame_results=[]; frame_sig=[]; frame_exit_center=[]

def reset_track():
    global trk_box,trk_vel,trk_vel_prev,trk_coast,trk_vel_age,trk_coasting,vel_history
    trk_box=None; trk_vel=np.zeros(2); trk_vel_prev=np.zeros(2)
    trk_coast=0; trk_vel_age=0; trk_coasting=False; vel_history=[]

def reset_cand():
    global cand_box,cand_vel,cand_frames
    cand_box=None; cand_vel=np.zeros(2); cand_frames=0

for fi,dets in enumerate(filtered):
    is_pred=False; sig=0.0; chosen=None
    relink_old_c=None; relink_frame=False

    if trk_box is not None:
        spd=float(np.linalg.norm(trk_vel))
        if trk_coasting:
            chosen=pick_best_det(dets,trk_box,MAX_LINK_PX)
            if chosen is None: reset_track()
        else:
            thresh=spd*VEL_COAST_FACTOR if spd>2.0 else MAX_LINK_PX
            chosen=pick_best_det(dets,trk_box,thresh)
            if chosen is None:
                if trk_coast<MAX_COAST and spd>0.5:
                    chosen=pick_nearest(low_unique[fi]+filtered[fi],center(trk_box),MAX_LINK_PX)
                    if chosen is None:
                        trk_coasting=True
                        trk_vel=trk_vel*COAST_DECAY
                        trk_box=shift_box(trk_box,trk_vel)
                        trk_coast+=1; is_pred=True
                else:
                    reset_track()

    if trk_box is None and chosen is None:
        if exit_center is not None:
            nearest=pick_nearest(low_unique[fi]+filtered[fi],exit_center,EXIT_RELINK_PX)
            if nearest is not None:
                bc=center(nearest[:4])
                dist=float(np.linalg.norm(bc-np.array(exit_center)))
                relink_old_c=np.array(exit_center)
                relink_frame=True
                trk_box=nearest[:4]
                trk_vel=exit_vel.copy()
                trk_vel_prev=exit_vel.copy() if np.linalg.norm(exit_vel)>=MIN_VEL_FOR_HIT else np.zeros(2)
                trk_vel_age=3
                trk_coast=0; trk_coasting=False
                exit_center=None; chosen=nearest
                vel_history=[exit_vel.copy()]
                reset_cand()
                print(f"  f{fi:04d} EXIT RELINK ✓ dist={dist:.0f} conf={nearest[4]:.2f} exit_vel_norm={np.linalg.norm(exit_vel):.1f}")

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
                exit_center=None; vel_history=[cand_vel.copy()]; reset_cand()
                print(f"  f{fi:04d} NEW TRACK confirmed")
        elif chosen is None:
            reset_cand()

    if chosen is not None and not is_pred:
        new_c  = center(chosen[:4])
        old_c  = relink_old_c if relink_old_c is not None else center(trk_box)
        new_vel= new_c - old_c

        if trk_vel_age >= 3:
            sig, info = compute_sig(trk_vel_prev, new_vel, chosen[:4],
                                    ref_area, vel_history)
            if info["skip"]:
                print(f"    SIG f{fi:04d} SKIP {info['skip']}  area={info['area']} ratio={info['ratio']}")
            else:
                print(f"    SIG f{fi:04d} "
                      f"area={info['area']}({info['ratio']}) scale={info['scale']} "
                      f"|rv1|={info['rn1']} |rv2|={info['rn2']} real_acc={info['real_acc']} "
                      f"h1={info['h1']} h2={info['h2']} turn={info['turn']} turn_sig={info['turn_sig']} "
                      f"σ_hdg={info['sigma_hdg']} smooth={info['smooth']} "
                      f"→ sig={info['sig']}")

        if not relink_frame:
            trk_vel_prev=trk_vel.copy()
        trk_vel=new_vel
        trk_box=chosen[:4]
        trk_coast=0; trk_coasting=False; trk_vel_age+=1
        vel_history.append(new_vel.copy())
        if len(vel_history)>6: vel_history.pop(0)

        if is_near_edge(trk_box,W,H):
            ecx,ecy=clamp_to_edge(trk_box,W,H)
            exit_center=(ecx,ecy); exit_vel=trk_vel.copy()
            print(f"  f{fi:04d} EDGE ARM ({ecx:.0f},{ecy:.0f})")
        else:
            exit_center=None

    frame_results.append((list(trk_box) if trk_box is not None else None, is_pred))
    frame_sig.append(sig)
    frame_exit_center.append(tuple(exit_center) if exit_center is not None else None)

    if trk_box is not None:
        sig_str=f" sig={sig:.2f}" if sig>0 else ""
        print(f"  f{fi:04d} {'P' if is_pred else 'R'} "
              f"vel={np.linalg.norm(trk_vel):.1f} age={trk_vel_age}{sig_str}")

# ── hit detection ─────────────────────────────────────────────────────────────
print("Hit detection...")
sig_max=max(frame_sig) if max(frame_sig)>0 else 1.0
print(f"  MIN_SIG_THRESH={MIN_SIG_THRESH}  sig_max={sig_max:.2f}")
hit_frames=set()
in_cluster=False
for i,s in enumerate(frame_sig):
    if s>=MIN_SIG_THRESH:
        if not in_cluster:
            hit_frames.add(i)
            in_cluster=True
            print(f"  cluster start f{i:04d} sig={s:.2f}")
    else:
        if in_cluster:
            print(f"  cluster end   f{i-1:04d}")
        in_cluster=False
print(f"  {len(hit_frames)} hit candidates: {sorted(hit_frames)}")

# ── pass 2: render ────────────────────────────────────────────────────────────
print("Pass 2: rendering...")
cap=cv2.VideoCapture(VIDEO_PATH)
writer=cv2.VideoWriter(OUTPUT_PATH,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
trail=[]

fi=0
while True:
    ret,frame=cap.read()
    if not ret: break

    # 全帧常驻显示所有 low_unique（橙）和 filtered（青绿）
    for d in low_unique[fi] if fi<len(low_unique) else []:
        x1,y1,x2,y2=map(int,d[:4])
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,100,0),1)
        cv2.putText(frame,f"{d[4]:.2f}",(x1,max(0,y1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,100,0),1)
    for d in filtered[fi] if fi<len(filtered) else []:
        x1,y1,x2,y2=map(int,d[:4])
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,100),1)
        cv2.putText(frame,f"{d[4]:.2f}",(x1,max(0,y1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,200,100),1)

    box,is_pred=frame_results[fi] if fi<len(frame_results) else (None,False)
    if box is not None:
        x1,y1,x2,y2=map(int,box); cx,cy=(x1+x2)//2,(y1+y2)//2
        trail.append((cx,cy,is_pred))
        if len(trail)>TRAIL_LEN: trail.pop(0)
        color=(180,180,180) if is_pred else (0,255,255)
        lw=2
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,lw)
        cv2.putText(frame,"pred" if is_pred else "ball",(x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,lw)
        cv2.circle(frame,(cx,cy),4,color,-1)

    for i in range(1,len(trail)):
        alpha=i/len(trail)
        c=(0,int(255*alpha),int(255*(1-alpha)))
        cv2.line(frame,trail[i-1][:2],trail[i][:2],c,1 if trail[i][2] else 2)

    # sig bar（原始值，未归一化，体现实时能量）
    BAR_W=280
    raw_sig=frame_sig[fi] if fi<len(frame_sig) else 0.0
    bar_fill=min(raw_sig/max(sig_max,1.0),1.0)
    x0=W-BAR_W-15
    cv2.rectangle(frame,(x0,15),(x0+BAR_W,38),(30,30,30),-1)
    bar_color=(0,80,255) if raw_sig>=MIN_SIG_THRESH else (0,200,255)
    cv2.rectangle(frame,(x0,15),(x0+int(bar_fill*BAR_W),38),bar_color,-1)
    cv2.putText(frame,f"sig:{raw_sig:.2f} thresh:{MIN_SIG_THRESH}",(x0,12),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,200,255),1)

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