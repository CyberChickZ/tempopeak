"""
ball_finder_v14c.py — A×B, pred frames included
sig = A × B
  A = clip(real_acc / A_max, 0, 1)   透视归一化加速度幅度
  B = sin²(Δheading/2)               转向信号，t-2→t-1 vs t-1→t，无smooth
  pred 帧也参与 sig 计算（用 trk_vel_prev, trk_vel）
  A_max = max_real_vel(99th) * 2.1   pass A 统计
  ref_area = (p95 + p5*0.4) / 2
hit: sig >= MIN_SIG_THRESH 的连续簇，窗口内取 sig 最大的帧
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

VIDEO_PATH  = "/Users/harryzhang/git/tempopeak/datasets/tenis-backview/video2.mp4"
# VIDEO_PATH  = "/Users/harryzhang/git/tempopeak/datasets/serve/00013.mp4"

OUTPUT_PATH = "/Users/harryzhang/git/tempopeak/outputs/ball_finder_v14c.mp4"
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
MIN_CONFIRM_VEL      = 8.0
EXIT_RELINK_PX       = 200
EDGE_ARM_MARGIN      = 80
MIN_VEL_FOR_SIG      = 10.0
MIN_AREA_RATIO       = 0.15
MIN_SIG_THRESH       = 0.08
MERGE_WINDOW         = 2
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
    return float(np.degrees(np.arctan2(v[0], -v[1])) % 360.0)

def min_turn_deg(v1, v2):
    if np.linalg.norm(v1)<1e-9 or np.linalg.norm(v2)<1e-9: return 0.0
    delta=((heading(v2)-heading(v1)+180)%360)-180
    return abs(delta)

def pick_best_det(dets, ref_box, max_dist):
    if not dets: return None
    rc=center(ref_box)
    cands=[(d,float(np.linalg.norm(center(d[:4])-rc))) for d in dets]
    cands=[(d,dist) for d,dist in cands if dist<=max_dist]
    if not cands: return None
    return min(cands, key=lambda x: x[1]-squareness(x[0][:4])*5)[0]

def pick_nearest(dets, ref_center, max_dist):
    if not dets: return None
    rc=np.array(ref_center)
    cands=[(d,float(np.linalg.norm(center(d[:4])-rc))) for d in dets]
    cands=[(d,dist) for d,dist in cands if dist<=max_dist]
    if not cands: return None
    return min(cands, key=lambda x: x[1])[0]

def compute_sig(v1, v2, box, ref_area, A_max):
    area  = box_area(box)
    ratio = area / max(ref_area, 1.0)
    if ratio < MIN_AREA_RATIO:
        return 0.0, 0.0, 0.0, {"skip": f"area_ratio={ratio:.2f}<{MIN_AREA_RATIO}",
                                "area": f"{area:.0f}"}
    scale    = float(np.sqrt(ref_area / max(area, 1.0)))
    rv2      = v2 * scale
    rn2      = float(np.linalg.norm(rv2))
    if rn2 < MIN_VEL_FOR_SIG:
        return 0.0, 0.0, 0.0, {"skip": f"|rv2|={rn2:.1f}<{MIN_VEL_FOR_SIG}",
                                "area": f"{area:.0f}", "ratio": f"{ratio:.2f}",
                                "scale": f"{scale:.2f}", "rn2": f"{rn2:.1f}"}
    rv1      = v1 * scale
    rn1      = float(np.linalg.norm(rv1))
    real_acc = float(np.linalg.norm(rv2 - rv1))
    A        = float(np.clip(real_acc / max(A_max, 1.0), 0.0, 1.0))
    turn_d   = min_turn_deg(v1, v2)
    B        = float((1.0 - np.cos(np.radians(turn_d))) / 2.0)
    sig      = A * B
    info = {
        "skip"    : None,
        "area"    : f"{area:.0f}",
        "ratio"   : f"{ratio:.2f}",
        "scale"   : f"{scale:.2f}",
        "rn1"     : f"{rn1:.1f}",
        "rn2"     : f"{rn2:.1f}",
        "real_acc": f"{real_acc:.1f}",
        "A"       : f"{A:.4f}",
        "h1"      : f"{heading(v1):.1f}°",
        "h2"      : f"{heading(v2):.1f}°",
        "turn"    : f"{turn_d:.1f}°",
        "B"       : f"{B:.4f}",
        "sig"     : f"{sig:.4f}",
    }
    return sig, A, B, info

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
    unique=[d for d in raw_low[fi]
            if not any(boxes_overlap(d[:4],h[:4]) for h in raw[fi])]
    low_unique.append(unique)

# ── area stats ────────────────────────────────────────────────────────────────
all_areas=[box_area(d[:4]) for frame in raw for d in frame if box_area(d[:4])>4]
if all_areas:
    p95=float(np.percentile(all_areas,95)); p5=float(np.percentile(all_areas,5))
    ref_area=(p95+p5*0.4)/2.0; min_track_area=p95*0.15
else:
    p95=400; p5=50; ref_area=220; min_track_area=60
print(f"  area p5={p5:.0f} p95={p95:.0f} → ref={ref_area:.0f} min_track={min_track_area:.0f}px²")

for fi in range(len(filtered)):
    filtered[fi]=[d for d in filtered[fi] if box_area(d[:4])>=min_track_area]
for fi in range(len(low_unique)):
    low_unique[fi]=[d for d in low_unique[fi] if box_area(d[:4])>=min_track_area]

# ── pass A: 收集 real_vel ─────────────────────────────────────────────────────
print("Pass A: collecting vel stats...")
trk_box=None; trk_vel=np.zeros(2); trk_coast=0; trk_coasting=False
cand_box=None; cand_vel=np.zeros(2); cand_frames=0; cand_vels=[]
all_real_vels=[]

for fi,dets in enumerate(filtered):
    chosen=None; is_pred=False
    if trk_box is not None:
        spd=float(np.linalg.norm(trk_vel))
        if trk_coasting:
            chosen=pick_best_det(dets,trk_box,MAX_LINK_PX)
            if chosen is None: trk_box=None; trk_vel=np.zeros(2); trk_coast=0; trk_coasting=False
        else:
            thresh=spd*VEL_COAST_FACTOR if spd>2.0 else MAX_LINK_PX
            chosen=pick_best_det(dets,trk_box,thresh)
            if chosen is None:
                if trk_coast<MAX_COAST and spd>0.5:
                    chosen=pick_nearest(low_unique[fi]+dets,center(trk_box),MAX_LINK_PX)
                    if chosen is None:
                        trk_coasting=True; trk_vel=trk_vel*COAST_DECAY
                        trk_box=shift_box(trk_box,trk_vel); trk_coast+=1; is_pred=True
                else:
                    trk_box=None; trk_vel=np.zeros(2); trk_coast=0; trk_coasting=False
    if trk_box is None and chosen is None and dets:
        best=max(dets,key=lambda d:d[4]); bc=center(best[:4])
        if cand_box is not None:
            if float(np.linalg.norm(bc-center(cand_box)))<=MAX_LINK_PX:
                cand_vel=bc-center(cand_box); cand_box=best[:4]; cand_frames+=1
                cand_vels.append(float(np.linalg.norm(cand_vel)))
            else:
                cand_box=best[:4]; cand_vel=np.zeros(2); cand_frames=1; cand_vels=[0.0]
        else:
            cand_box=best[:4]; cand_vel=np.zeros(2); cand_frames=1; cand_vels=[0.0]
        if cand_frames>=NEW_TRACK_MIN_FRAMES:
            avg=float(np.mean(cand_vels)) if cand_vels else 0.0
            if avg>=MIN_CONFIRM_VEL:
                trk_box=cand_box; trk_vel=cand_vel
            cand_box=None; cand_vel=np.zeros(2); cand_frames=0; cand_vels=[]
    if chosen is not None and not is_pred:
        new_c=center(chosen[:4]); old_c=center(trk_box) if trk_box is not None else new_c
        new_vel=new_c-old_c
        scale=float(np.sqrt(ref_area/max(box_area(chosen[:4]),1.0)))
        all_real_vels.append(float(np.linalg.norm(new_vel*scale)))
        trk_vel=new_vel; trk_box=chosen[:4]; trk_coast=0; trk_coasting=False

if all_real_vels:
    max_real_vel=float(np.percentile(all_real_vels,99))
    A_max=max_real_vel*2.1
else:
    max_real_vel=50.0; A_max=105.0
print(f"  max_real_vel(99th)={max_real_vel:.1f}  A_max={A_max:.1f}")

# ── pass B: 正式 tracking ─────────────────────────────────────────────────────
print("Tracking...")
trk_box=None; trk_vel=np.zeros(2); trk_vel_prev=np.zeros(2)
trk_coast=0; trk_vel_age=0; trk_coasting=False
exit_center=None; exit_vel=np.zeros(2)
cand_box=None; cand_vel=np.zeros(2); cand_frames=0; cand_vels=[]

frame_results=[]; frame_sig=[]; frame_AB=[]

for fi,dets in enumerate(filtered):
    is_pred=False; sig=0.0; A_v=0.0; B_v=0.0; chosen=None
    relink_old_c=None; relink_frame=False

    # ── confirmed track ───────────────────────────────────────────────────
    if trk_box is not None:
        spd=float(np.linalg.norm(trk_vel))
        if trk_coasting:
            chosen=pick_best_det(dets,trk_box,MAX_LINK_PX)
            if chosen is None:
                trk_box=None; trk_vel=np.zeros(2); trk_vel_prev=np.zeros(2)
                trk_coast=0; trk_vel_age=0; trk_coasting=False
                print(f"  f{fi:04d} LOST (coasting)")
        else:
            thresh=spd*VEL_COAST_FACTOR if spd>2.0 else MAX_LINK_PX
            chosen=pick_best_det(dets,trk_box,thresh)
            if chosen is None:
                if trk_coast<MAX_COAST and spd>0.5:
                    chosen=pick_nearest(low_unique[fi]+dets,center(trk_box),MAX_LINK_PX)
                    if chosen is None:
                        trk_coasting=True; trk_vel=trk_vel*COAST_DECAY
                        trk_box=shift_box(trk_box,trk_vel)
                        trk_coast+=1; is_pred=True
                else:
                    trk_box=None; trk_vel=np.zeros(2); trk_vel_prev=np.zeros(2)
                    trk_coast=0; trk_vel_age=0; trk_coasting=False
                    print(f"  f{fi:04d} LOST")

    # ── exit relink + candidate ───────────────────────────────────────────
    if trk_box is None and chosen is None:
        if exit_center is not None:
            nearest=pick_nearest(low_unique[fi]+dets,exit_center,EXIT_RELINK_PX)
            if nearest is not None:
                dist=float(np.linalg.norm(center(nearest[:4])-np.array(exit_center)))
                relink_old_c=np.array(exit_center); relink_frame=True
                trk_box=nearest[:4]; trk_vel=exit_vel.copy()
                trk_vel_prev=exit_vel.copy() if np.linalg.norm(exit_vel)>=MIN_VEL_FOR_SIG else np.zeros(2)
                trk_vel_age=3; trk_coast=0; trk_coasting=False
                exit_center=None; chosen=nearest
                cand_box=None; cand_vel=np.zeros(2); cand_frames=0; cand_vels=[]
                print(f"  f{fi:04d} EXIT RELINK ✓ dist={dist:.0f} conf={nearest[4]:.2f} exit_vel={np.linalg.norm(exit_vel):.1f}")

        if chosen is None and dets:
            best=max(dets,key=lambda d:d[4]); bc=center(best[:4])
            if cand_box is not None:
                if float(np.linalg.norm(bc-center(cand_box)))<=MAX_LINK_PX:
                    cand_vel=bc-center(cand_box); cand_box=best[:4]; cand_frames+=1
                    cand_vels.append(float(np.linalg.norm(cand_vel)))
                else:
                    cand_box=best[:4]; cand_vel=np.zeros(2); cand_frames=1; cand_vels=[0.0]
            else:
                cand_box=best[:4]; cand_vel=np.zeros(2); cand_frames=1; cand_vels=[0.0]
            if cand_frames>=NEW_TRACK_MIN_FRAMES:
                avg=float(np.mean(cand_vels)) if cand_vels else 0.0
                if avg>=MIN_CONFIRM_VEL:
                    trk_box=cand_box; trk_vel=cand_vel; trk_vel_age=cand_frames
                    trk_coast=0; trk_coasting=False; chosen=best; exit_center=None
                    print(f"  f{fi:04d} NEW TRACK confirmed avg_vel={avg:.1f}")
                else:
                    print(f"  f{fi:04d} CAND rejected avg_vel={avg:.1f}<{MIN_CONFIRM_VEL}")
                cand_box=None; cand_vel=np.zeros(2); cand_frames=0; cand_vels=[]
        elif chosen is None:
            cand_box=None; cand_vel=np.zeros(2); cand_frames=0; cand_vels=[]

    # ── update + sig ──────────────────────────────────────────────────────
    if chosen is not None and not is_pred:
        new_c  =center(chosen[:4])
        old_c  =relink_old_c if relink_old_c is not None else center(trk_box)
        new_vel=new_c-old_c

        if trk_vel_age>=3:
            sig,A_v,B_v,info=compute_sig(trk_vel_prev,new_vel,chosen[:4],ref_area,A_max)
            if info["skip"]:
                print(f"    SIG f{fi:04d} SKIP {info['skip']}")
            else:
                flag=" *** HIT ***" if sig>=MIN_SIG_THRESH else ""
                print(f"    SIG f{fi:04d} "
                      f"area={info['area']}(r={info['ratio']}) sc={info['scale']} "
                      f"|rv1|={info['rn1']} |rv2|={info['rn2']} "
                      f"real_acc={info['real_acc']} A={info['A']} "
                      f"h1={info['h1']} h2={info['h2']} turn={info['turn']} B={info['B']} "
                      f"→ sig={info['sig']}{flag}")

        if not relink_frame:
            trk_vel_prev=trk_vel.copy()
        trk_vel=new_vel; trk_box=chosen[:4]
        trk_coast=0; trk_coasting=False; trk_vel_age+=1

        if is_near_edge(trk_box,W,H):
            ecx,ecy=clamp_to_edge(trk_box,W,H)
            exit_center=(ecx,ecy); exit_vel=trk_vel.copy()
            print(f"  f{fi:04d} EDGE ARM ({ecx:.0f},{ecy:.0f})")
        else:
            exit_center=None

    elif is_pred and trk_box is not None and trk_vel_age>=3:
        # pred 帧：用 trk_vel_prev→trk_vel 算 sig，box 用预测位置
        sig,A_v,B_v,info=compute_sig(trk_vel_prev,trk_vel,trk_box,ref_area,A_max)
        if info["skip"]:
            print(f"    SIG f{fi:04d} PRED SKIP {info['skip']}")
        else:
            flag=" *** HIT ***" if sig>=MIN_SIG_THRESH else ""
            print(f"    SIG f{fi:04d} PRED "
                  f"area={info['area']}(r={info['ratio']}) sc={info['scale']} "
                  f"|rv1|={info['rn1']} |rv2|={info['rn2']} "
                  f"real_acc={info['real_acc']} A={info['A']} "
                  f"h1={info['h1']} h2={info['h2']} turn={info['turn']} B={info['B']} "
                  f"→ sig={info['sig']}{flag}")

    frame_results.append((list(trk_box) if trk_box is not None else None, is_pred))
    frame_sig.append(sig); frame_AB.append((A_v,B_v))

    if trk_box is not None:
        sig_str=f" sig={sig:.4f}" if sig>0 else ""
        print(f"  f{fi:04d} {'P' if is_pred else 'R'} "
              f"vel={np.linalg.norm(trk_vel):.1f} age={trk_vel_age}{sig_str}")

# ── hit detection ─────────────────────────────────────────────────────────────
print("Hit detection...")
sig_max=max(frame_sig) if frame_sig and max(frame_sig)>0 else 1.0
print(f"  A_max={A_max:.1f}  MIN_SIG_THRESH={MIN_SIG_THRESH}  sig_max={sig_max:.4f}")

raw_hits=[]; in_cluster=False
for i,s in enumerate(frame_sig):
    if s>=MIN_SIG_THRESH:
        if not in_cluster:
            raw_hits.append(i); in_cluster=True
            print(f"  cluster start f{i:04d} sig={s:.4f}")
    else:
        if in_cluster: print(f"  cluster end   f{i-1:04d}")
        in_cluster=False

hit_frames=set(); i=0
while i<len(raw_hits):
    group=[raw_hits[i]]; j=i+1
    while j<len(raw_hits) and raw_hits[j]-group[-1]<=MERGE_WINDOW:
        group.append(raw_hits[j]); j+=1
    best=max(group, key=lambda f: frame_sig[f])
    hit_frames.add(best)
    print(f"  group {group} → keep f{best:04d} sig={frame_sig[best]:.4f}")
    i=j
print(f"  {len(hit_frames)} hit candidates: {sorted(hit_frames)}")

# ── pass 2: render ────────────────────────────────────────────────────────────
print("Pass 2: rendering...")
cap=cv2.VideoCapture(VIDEO_PATH)
writer=cv2.VideoWriter(OUTPUT_PATH,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
trail=[]

BAR_W=220; BAR_H=16; BAR_X=W-BAR_W-15; ROW_H=24

def draw_bar(frame, row, val, label, fmt, color, thresh_ratio=None):
    y=12+row*ROW_H
    fill=int(min(max(val,0.0),1.0)*BAR_W)
    cv2.rectangle(frame,(BAR_X,y),(BAR_X+BAR_W,y+BAR_H),(40,40,40),-1)
    cv2.rectangle(frame,(BAR_X,y),(BAR_X+fill,y+BAR_H),color,-1)
    if thresh_ratio is not None:
        tx=BAR_X+int(thresh_ratio*BAR_W)
        cv2.line(frame,(tx,y),(tx,y+BAR_H),(0,255,255),1)
    cv2.putText(frame,f"{label}: {fmt}",(BAR_X-200,y+BAR_H-3),
                cv2.FONT_HERSHEY_SIMPLEX,0.40,(220,220,220),1)

fi=0
while True:
    ret,frame=cap.read()
    if not ret: break

    for d in (low_unique[fi] if fi<len(low_unique) else []):
        x1,y1,x2,y2=map(int,d[:4])
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,100,0),1)
        cv2.putText(frame,f"{d[4]:.2f}",(x1,max(0,y1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.33,(255,100,0),1)
    for d in (filtered[fi] if fi<len(filtered) else []):
        x1,y1,x2,y2=map(int,d[:4])
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,100),1)
        cv2.putText(frame,f"{d[4]:.2f}",(x1,max(0,y1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.33,(0,200,100),1)

    box,is_pred=frame_results[fi] if fi<len(frame_results) else (None,False)
    if box is not None:
        x1,y1,x2,y2=map(int,box); cx,cy=(x1+x2)//2,(y1+y2)//2
        trail.append((cx,cy,is_pred))
        if len(trail)>TRAIL_LEN: trail.pop(0)
        col=(180,180,180) if is_pred else (0,255,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
        cv2.putText(frame,"pred" if is_pred else "ball",(x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,col,2)
        cv2.circle(frame,(cx,cy),4,col,-1)

    for i in range(1,len(trail)):
        alpha=i/len(trail)
        c=(0,int(255*alpha),int(255*(1-alpha)))
        cv2.line(frame,trail[i-1][:2],trail[i][:2],c,1 if trail[i][2] else 2)

    A_v,B_v=frame_AB[fi] if fi<len(frame_AB) else (0.0,0.0)
    s_v=frame_sig[fi] if fi<len(frame_sig) else 0.0

    draw_bar(frame,0, A_v,        "A acc",  f"{A_v:.4f}", (100,220,255))
    draw_bar(frame,1, B_v,        "B turn", f"{B_v:.4f}", (60,180,255))
    draw_bar(frame,2, min(s_v,1.0),"S=A×B", f"{s_v:.4f}",
             (0,60,255) if s_v>=MIN_SIG_THRESH else (0,200,255),
             thresh_ratio=MIN_SIG_THRESH)

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