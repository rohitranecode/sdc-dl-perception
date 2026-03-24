"""
Bird's Eye View + HUD Overlay — DL Perception System
"""
import cv2
import numpy as np
import time

_FPS_PREV = time.time()
_FPS_CNT  = 0
_FPS_SMTH = 30.0


def get_fps():
    global _FPS_PREV, _FPS_CNT, _FPS_SMTH
    now = time.time()
    _FPS_CNT += 1
    dt = now - _FPS_PREV
    if dt >= 0.5:
        raw = _FPS_CNT / dt
        _FPS_SMTH = 0.75 * _FPS_SMTH + 0.25 * raw
        _FPS_PREV = now
        _FPS_CNT  = 0
    return _FPS_SMTH


def warp_bev(frame, horizon_frac=0.60):
    h, w = frame.shape[:2]
    hy   = int(h * horizon_frac)
    tilt = 0.08
    src  = np.float32([
        [int(w*(0.5-tilt)), hy], [int(w*(0.5+tilt)), hy],
        [int(w*0.95), h-10],     [int(w*0.05), h-10],
    ])
    dst  = np.float32([
        [int(w*0.25),0],[int(w*0.75),0],
        [int(w*0.75),h],[int(w*0.25),h],
    ])
    M   = cv2.getPerspectiveTransform(src, dst)
    bev = cv2.warpPerspective(frame, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderValue=(20,20,20))
    return bev, M


def project_lanes_bev(bev_img, lanes, M):
    """Project polynomial lane points into BEV space."""
    vis = bev_img.copy()
    overlay = bev_img.copy()
    h, w = vis.shape[:2]

    LANE_COLORS = [(0,255,120),(0,200,255),(255,165,0),(255,80,80)]
    bev_lanes = []

    for i, lane_pts in enumerate(lanes[:4]):
        if len(lane_pts) < 2:
            continue
        col   = LANE_COLORS[i % len(LANE_COLORS)]
        pts_h = lane_pts.reshape(-1, 1, 2).astype(np.float32)
        dst   = cv2.perspectiveTransform(pts_h, M)
        if dst is None:
            continue
        dst = dst.reshape(-1, 2).astype(np.int32)
        dst = dst[(dst[:,0]>=0)&(dst[:,0]<w)&(dst[:,1]>=0)&(dst[:,1]<h)]
        if len(dst) > 1:
            cv2.polylines(vis, [dst], False, col, 3, cv2.LINE_AA)
        bev_lanes.append(dst)

    # Fill ego lane
    if len(bev_lanes) >= 2:
        left, right = bev_lanes[0], bev_lanes[1]
        if len(left)>1 and len(right)>1:
            poly = np.vstack([left, right[::-1]])
            cv2.fillPoly(overlay, [poly], (0,180,90))
    vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
    cv2.putText(vis,"BEV",(6,18),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,210,255),1)
    return vis


def occupancy_map(detections, frame_shape, size=(400,400)):
    mh, mw = size
    occ    = np.zeros((mh, mw, 3), dtype=np.uint8)
    H, W   = frame_shape[:2]

    for gx in range(0, mw, mw//6):
        cv2.line(occ,(gx,0),(gx,mh),(28,28,28),1)
    for gy in range(0, mh, mh//8):
        cv2.line(occ,(0,gy),(mw,gy),(28,28,28),1)

    ego_x,ego_y = mw//2, mh-35
    cv2.rectangle(occ,(ego_x-12,ego_y-36),(ego_x+12,ego_y),(0,200,255),-1)
    cv2.putText(occ,"EGO",(ego_x-13,ego_y-14),cv2.FONT_HERSHEY_SIMPLEX,0.32,(0,0,0),1)
    cv2.arrowedLine(occ,(ego_x,ego_y-38),(ego_x,ego_y-58),(0,200,255),2,tipLength=0.4)

    for d_m in [20,40,60,80]:
        r = int(d_m/120*mh)
        cv2.circle(occ,(ego_x,ego_y),r,(45,80,45),1,cv2.LINE_AA)
        cv2.putText(occ,f"{d_m}m",(ego_x+r+2,ego_y),cv2.FONT_HERSHEY_SIMPLEX,0.28,(55,120,55),1)

    COLS = {"car":(0,200,255),"truck":(255,140,0),"bus":(180,0,255),
            "person":(80,80,255),"motorcycle":(0,255,180),"bicycle":(120,255,80),
            "traffic light":(255,220,0),"stop sign":(0,60,220)}

    for det in detections:
        x1,y1,x2,y2 = det["bbox"]
        dist = det["distance"]
        cls  = det["class"]
        lat  = ((x1+x2)/2 - W/2) / (W/2)
        bx   = int(ego_x + lat*mw*0.38)
        by   = int(ego_y - (dist/120)*(mh-50))
        by   = max(5, min(mh-5, by))
        col  = COLS.get(cls,(180,180,180))
        bw   = max(7,int(24*(1-dist/120)))
        bh   = max(5,int(16*(1-dist/120)))
        cv2.rectangle(occ,(bx-bw//2,by-bh),(bx+bw//2,by),col,-1)
        cv2.putText(occ,cls[:3].upper(),(bx-10,by-bh-2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.26,col,1)

    cv2.rectangle(occ,(0,0),(mw-1,mh-1),(70,70,70),1)
    cv2.putText(occ,"OCCUPANCY GRID",(4,12),cv2.FONT_HERSHEY_SIMPLEX,0.32,(130,130,130),1)
    return occ


def hud_overlay(frame, lane_metrics, detections, depth_stats,
                model_info=None, speed=65):
    vis = frame.copy()
    h, w = vis.shape[:2]

    panel = vis.copy()
    cv2.rectangle(panel,(0,0),(w,56),(0,0,0),-1)
    cv2.rectangle(panel,(0,h-48),(w,h),(0,0,0),-1)
    vis = cv2.addWeighted(vis,0.55,panel,0.45,0)

    # Title
    cv2.putText(vis,"DL PERCEPTION v3.0  |  YOLOv8 - MiDaS - SegFormer - LaneNet",
                (10,20),cv2.FONT_HERSHEY_DUPLEX,0.48,(0,210,255),1,cv2.LINE_AA)

    # Model mode badges - ASCII only, no unicode
    if model_info:
        x_off = 10
        for name, loaded in model_info.items():
            col   = (0,220,100) if loaded else (0,180,255)
            mode  = "ONNX" if loaded else "SIM"
            label = f"[{name}:{mode}]"
            (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
            cv2.putText(vis, label, (x_off, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1)
            x_off += lw + 8

    fps = get_fps()
    fps_col = (0,255,100) if fps>15 else (0,200,255) if fps>8 else (0,80,255)
    cv2.putText(vis,f"FPS:{fps:.0f}",(w-95,20),cv2.FONT_HERSHEY_DUPLEX,0.48,fps_col,1)

    lm   = lane_metrics or {}
    conf = float(lm.get("confidence",0))
    cc   = (0,255,120) if conf>0.7 else (0,200,255) if conf>0.4 else (0,80,255)
    for i,(k,v,c) in enumerate([
        ("LANES","",          (0,210,255)),
        ("Conf", f"{conf:.0%}", cc),
        ("Width",lm.get("lane_width","N/A"),(210,210,210)),
        ("Offs", lm.get("offset","N/A"),    (210,210,210)),
    ]):
        cv2.putText(vis,f"{k}: {v}",(10,80+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.38,c,1)

    dets    = detections or []
    nearest = dets[0]["distance"] if dets else None
    nc      = (0,80,255) if nearest and nearest<20 else (210,210,210)
    for i,(k,v,c) in enumerate([
        ("OBJECTS","",           (0,210,255)),
        ("Count",  str(len(dets)),(210,210,210)),
        ("Nearest",f"{nearest:.0f}m" if nearest else "CLEAR", nc),
    ]):
        cv2.putText(vis,f"{k}: {v}",(w-160,80+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.38,c,1)

    dc = (depth_stats or {}).get("center_dist","N/A")
    cv2.putText(vis,f"DEPTH CENTER: {dc}",(12,h-12),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(150,150,150),1)
    cv2.putText(vis,f"{speed} km/h",(w//2-38,h-12),
                cv2.FONT_HERSHEY_DUPLEX,0.65,(0,210,255),1)

    if nearest and nearest < 15:
        cv2.putText(vis,"!! COLLISION WARNING !!",(w//2-155,h-32),
                    cv2.FONT_HERSHEY_DUPLEX,0.72,(0,0,255),2)
    elif nearest and nearest < 30:
        cv2.putText(vis,"  CAUTION",(w//2-60,h-32),
                    cv2.FONT_HERSHEY_DUPLEX,0.60,(0,165,255),1)

    cx,cy = w//2, h//2
    cv2.line(vis,(cx-18,cy),(cx+18,cy),(0,255,200),1)
    cv2.line(vis,(cx,cy-18),(cx,cy+18),(0,255,200),1)
    cv2.circle(vis,(cx,cy),4,(0,255,200),1)
    return vis


def pipeline_grid(raw, lane_ann, obj_ann, depth_col, seg_ann, bev_ann, occ):
    th, tw = 240, 426
    def lab(img, txt):
        out = cv2.resize(img,(tw,th))
        cv2.rectangle(out,(0,0),(tw,20),(0,0,0),-1)
        cv2.putText(out,txt,(4,14),cv2.FONT_HERSHEY_DUPLEX,0.42,(0,210,255),1)
        return out
    imgs = [
        lab(raw,       "RAW INPUT"),
        lab(lane_ann,  "LANENET-UFLD"),
        lab(obj_ann,   "YOLOv8 DET"),
        lab(depth_col, "MiDaS DEPTH"),
        lab(seg_ann,   "SEGFORMER"),
        lab(bev_ann,   "BIRD EYE VIEW"),
    ]
    r1 = np.hstack(imgs[:3])
    r2 = np.hstack(imgs[3:])
    return np.vstack([r1,r2])
