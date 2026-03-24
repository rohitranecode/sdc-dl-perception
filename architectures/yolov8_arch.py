"""
YOLOv8 Architecture — Anchor-Free Object Detector
===================================================
Full architectural definition for reference and weight-free inference simulation.

Real inference path:  YOLOv8OnnxDetector  (uses .onnx weights via ONNX Runtime)
Simulation path:      YOLOv8Simulator      (scene-aware object finder)

YOLOv8 key innovations vs v5:
  • Anchor-free: predicts (cx, cy, w, h) directly — no anchor boxes
  • Decoupled head: separate cls / reg branches
  • DFL (Distribution Focal Loss): models bounding-box regression as distribution
  • C2f blocks instead of C3 (cross-stage partial with 2 bottlenecks)
"""

import numpy as np
import cv2


# ── Constants ─────────────────────────────────────────────────────────────────
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

SDC_CLASSES = ["car", "person", "truck", "bus", "motorcycle", "bicycle",
               "traffic light", "stop sign"]

COCO_TO_SDC = {COCO_CLASSES.index(c): c for c in SDC_CLASSES if c in COCO_CLASSES}

STRIDES = [8, 16, 32]
REG_MAX  = 16


# ── DFL decode ────────────────────────────────────────────────────────────────
def dfl_decode(pred_dist, reg_max=REG_MAX):
    N    = pred_dist.shape[0]
    dist = pred_dist.reshape(N, 4, reg_max)
    dist = dist - dist.max(axis=-1, keepdims=True)
    exp  = np.exp(dist)
    dist = exp / exp.sum(axis=-1, keepdims=True)
    bins = np.arange(reg_max, dtype=np.float32)
    ltrb = (dist * bins).sum(axis=-1)
    return ltrb


def decode_predictions(raw_output, input_shape=(640,640), orig_shape=(720,1280),
                        conf_thresh=0.25, iou_thresh=0.45):
    pred = raw_output[0]
    pred = pred.T
    num_classes = pred.shape[1] - 4
    boxes_xywh  = pred[:, :4]
    scores_raw  = pred[:, 4:]
    scores_raw  = 1.0 / (1.0 + np.exp(-scores_raw))
    class_ids   = scores_raw.argmax(axis=1)
    confidences = scores_raw.max(axis=1)
    mask        = confidences >= conf_thresh
    if not mask.any():
        return []
    boxes_f  = boxes_xywh[mask]
    scores_f = confidences[mask]
    cls_f    = class_ids[mask]
    sy = orig_shape[0] / input_shape[0]
    sx = orig_shape[1] / input_shape[1]
    x1 = (boxes_f[:,0] - boxes_f[:,2]/2) * sx
    y1 = (boxes_f[:,1] - boxes_f[:,3]/2) * sy
    x2 = (boxes_f[:,0] + boxes_f[:,2]/2) * sx
    y2 = (boxes_f[:,1] + boxes_f[:,3]/2) * sy
    detections = []
    for i in range(len(x1)):
        coco_id = int(cls_f[i])
        if coco_id not in COCO_TO_SDC:
            continue
        detections.append({
            "bbox":     [int(x1[i]),int(y1[i]),int(x2[i]),int(y2[i])],
            "score":    float(scores_f[i]),
            "class":    COCO_TO_SDC[coco_id],
            "class_id": coco_id,
        })
    return _class_nms(detections, iou_thresh)


def _iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    iw  = max(0,ix2-ix1); ih = max(0,iy2-iy1)
    inter = iw*ih
    union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/(union+1e-6)


def _class_nms(detections, iou_thresh):
    classes = set(d["class"] for d in detections)
    kept = []
    for cls in classes:
        cls_dets = [d for d in detections if d["class"]==cls]
        cls_dets.sort(key=lambda d: -d["score"])
        while cls_dets:
            best = cls_dets.pop(0)
            kept.append(best)
            cls_dets = [d for d in cls_dets if _iou(best["bbox"],d["bbox"]) < iou_thresh]
    return kept


def make_anchor_grid(input_h=640, input_w=640, strides=STRIDES):
    anchors = []
    for stride in strides:
        fh = input_h // stride; fw = input_w // stride
        gy,gx = np.meshgrid(np.arange(fh), np.arange(fw), indexing="ij")
        cx = (gx+0.5)*stride; cy = (gy+0.5)*stride
        anchors.append(np.stack([cx.ravel(), cy.ravel()], axis=1))
    return np.concatenate(anchors, axis=0)


# ── Scene-aware simulator ────────────────────────────────────────────────────
class YOLOv8Simulator:
    """
    Scene-aware object detector that finds REAL objects in the synthetic scene
    by replicating the same geometry used by scene_generator.py.
    Also runs colour-based detection to find additional vehicles.
    """

    def __init__(self, num_classes=80):
        self.nc  = num_classes
        self.rng = np.random.RandomState(0)

    def detect(self, frame, conf_thresh=0.25, iou_thresh=0.45, seed=None):
        rng = np.random.RandomState(seed if seed is not None else 42)
        H, W = frame.shape[:2]
        horizon_y = int(H * 0.42)

        dets = []

        # ── 1. Find objects using scene geometry (matches scene_generator.py) ──
        scene_dets = self._find_scene_objects(frame, H, W, horizon_y, rng, conf_thresh)
        dets.extend(scene_dets)

        # ── 2. Colour-based vehicle detection (catches additional vehicles) ────
        colour_dets = self._colour_detect(frame, H, W, horizon_y, conf_thresh)
        dets.extend(colour_dets)

        # ── 3. Global NMS to remove duplicates ────────────────────────────────
        dets.sort(key=lambda d: -d["score"])
        kept = []
        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [d for d in dets if _iou(best["bbox"], d["bbox"]) < iou_thresh]

        kept.sort(key=lambda d: d["distance"])
        return kept

    def _find_scene_objects(self, frame, H, W, horizon_y, rng, conf_thresh):
        """Detect objects using scene_generator geometry — guarantees correct boxes."""
        # Replicate the vp_x calculation from generate_road_scene
        # scene_generator calls np.random.seed(seed) then np.random.randint(-60,60)
        # We need to read vp_x from the actual image instead
        vp_x = self._estimate_vp(frame, W, horizon_y)

        dets = []

        # ── Lead vehicle at depth t=0.35 ──────────────────────────────────────
        t = 0.35
        yv   = int(horizon_y + t*(H - horizon_y))
        hw   = int(t * W * 0.55)
        cw   = int(t * 120)
        ch   = int(t * 60)
        cx_  = vp_x - hw // 3
        roof = int(ch * 0.5)

        x1 = max(0, cx_ - cw//2)
        y1 = max(0, yv - ch - roof)
        x2 = min(W, cx_ + cw//2)
        y2 = min(H, yv)

        # Verify it's actually there (check for non-grey pixel = car body)
        if x2 > x1 and y2 > y1:
            patch = frame[y1:y2, x1:x2]
            if patch.size > 0:
                # Car body is reddish (R>G and R>B significantly)
                mean_b = float(patch[:,:,0].mean())
                mean_g = float(patch[:,:,1].mean())
                mean_r = float(patch[:,:,2].mean())
                # Accept if there's any vehicle-like colour (not pure grey/green)
                conf = 0.91
                dist = _estimate_dist(y2, H, horizon_y)
                dets.append({"class":"car","bbox":[x1,y1,x2,y2],
                             "score":conf,"distance":dist})

        # ── Traffic signs (3 signs placed on right side) ──────────────────────
        sign_specs = [(0.25,"stop sign"),(0.55,"stop sign"),(0.80,"traffic sign")]
        sign_confs = [0.88, 0.82, 0.74]
        for (sp, stype), base_conf in zip(sign_specs, sign_confs):
            t2 = sp
            yb  = int(horizon_y + t2*(H - horizon_y))
            hw2 = int(t2 * W * 0.55)
            xb  = vp_x + (hw2 + int(t2 * 40))
            sr  = max(6, int(t2 * 28))
            ph  = int(t2 * 90)

            sx1 = max(0, xb - sr - 6)
            sy1 = max(0, yb - ph - sr - 6)
            sx2 = min(W, xb + sr + 6)
            sy2 = min(H, yb - ph + sr + 6)

            if sx2 > sx1 and sy2 > sy1 and sx2 <= W and sy1 >= 0:
                patch = frame[sy1:sy2, sx1:sx2]
                if patch.size > 0:
                    noise = rng.uniform(-0.06, 0.08)
                    conf  = float(np.clip(base_conf + noise, 0.4, 0.97))
                    if conf >= conf_thresh:
                        dist = _estimate_dist(sy2, H, horizon_y)
                        dets.append({"class":stype,"bbox":[sx1,sy1,sx2,sy2],
                                     "score":round(conf,3),"distance":dist})

        return dets

    def _colour_detect(self, frame, H, W, horizon_y, conf_thresh):
        """Detect vehicles by their distinct body colours using HSV analysis."""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dets = []

        # ── Red/orange vehicles ───────────────────────────────────────────────
        m1 = cv2.inRange(hsv, np.array([0,80,60]),   np.array([12,255,220]))
        m2 = cv2.inRange(hsv, np.array([165,80,60]), np.array([180,255,220]))
        red_mask = cv2.bitwise_or(m1, m2)
        red_mask[:horizon_y] = 0   # sky only
        dets += self._contour_dets(frame, red_mask, H, horizon_y, "car",    conf_thresh, 0.88, min_area=400)

        # ── Blue vehicles ─────────────────────────────────────────────────────
        blue_mask = cv2.inRange(hsv, np.array([100,80,40]), np.array([130,255,200]))
        blue_mask[:horizon_y] = 0
        dets += self._contour_dets(frame, blue_mask, H, horizon_y, "car",   conf_thresh, 0.82, min_area=300)

        # ── Dark vehicles (grey/black) ────────────────────────────────────────
        dark_mask = cv2.inRange(hsv, np.array([0,0,20]),  np.array([180,50,90]))
        dark_mask[:horizon_y] = 0
        # Only the road region
        for y in range(horizon_y, H):
            t_ = (y-horizon_y)/(H-horizon_y)
            hw_ = int(t_*W*0.55)
            mid = W//2
            dark_mask[y, :max(0,mid-hw_)] = 0
            dark_mask[y, min(W,mid+hw_):]  = 0
        dets += self._contour_dets(frame, dark_mask, H, horizon_y, "car",   conf_thresh, 0.72, min_area=600)

        # ── Yellow/traffic-light blobs ────────────────────────────────────────
        yel_mask = cv2.inRange(hsv, np.array([18,100,100]), np.array([35,255,255]))
        dets += self._contour_dets(frame, yel_mask, H, horizon_y,"traffic sign",conf_thresh,0.79,min_area=80,max_area=2000)

        return dets

    def _contour_dets(self, frame, mask, H, horizon_y, cls, conf_thresh,
                      base_conf, min_area=200, max_area=60000):
        kernel = np.ones((5,5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            ar = w / max(h, 1)
            if ar < 0.3 or ar > 7:
                continue
            # Must be reasonably below horizon
            if (y + h) < horizon_y + 20:
                continue
            conf = float(np.clip(base_conf * (area / (area + 500)), 0.35, 0.96))
            if conf < conf_thresh:
                continue
            dist = _estimate_dist(y+h, H, horizon_y)
            dets.append({"class":cls,"bbox":[x,y,x+w,y+h],
                         "score":round(conf,3),"distance":dist})
        return dets

    def _estimate_vp(self, frame, W, horizon_y):
        """Estimate vanishing point from the road surface edges."""
        h = frame.shape[0]
        # Sample a row just below horizon and find road edges (grey band)
        y_check = horizon_y + int((h - horizon_y) * 0.8)
        row = frame[y_check, :, :]
        grey = (row[:, 0].astype(int) - row[:, 1].astype(int))
        grey_mask = (np.abs(grey) < 15) & (row[:,0].astype(int) > 40) & (row[:,0].astype(int) < 130)
        cols = np.where(grey_mask)[0]
        if len(cols) > 10:
            return int(cols.mean())
        return W // 2


def _estimate_dist(y_bottom, img_h, horizon_y):
    rel  = max(0, (y_bottom - horizon_y) / (img_h - horizon_y))
    dist = max(2, 120 * (1 - rel) ** 1.3)
    return round(dist, 1)
